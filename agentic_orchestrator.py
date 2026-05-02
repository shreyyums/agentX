#!/usr/bin/env python3
"""
Agentic Orchestrator (LangChain/OpenAI) for loan eligibility conversations.

Example:
    python agentic_orchestrator.py

Then type:
    Check my eligibility for a 50L loan.

Set OPENAI_API_KEY to let LangChain/OpenAI polish the final answer. Without an
API key, the deterministic orchestrator still runs and prints a local response.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


DEFAULT_INTEREST_RATE = 10.5
DEFAULT_TENURE_MONTHS = 60
MAX_FOI_RATIO = 0.50
MIN_CREDIT_SCORE = 700
MIN_AGE = 21
MAX_AGE_AT_TENURE_END = 60
POONAWALLA_MIN_CIBIL = 750
POONAWALLA_UNSECURED_ABROAD_CAP = 10_000_000
POONAWALLA_SECURED_ABROAD_CAP = 30_000_000
POONAWALLA_SECURED_DOMESTIC_CAP = 5_000_000
POONAWALLA_MIN_LOAN = 100_000
POONAWALLA_SUPPORTED_LOCATIONS = {
    "usa",
    "united states",
    "uk",
    "united kingdom",
    "ireland",
    "australia",
    "sweden",
    "spain",
    "france",
    "germany",
    "new zealand",
    "singapore",
    "uae",
    "philippines",
    "canada",
    "india",
}
SELECT_UNIVERSITIES = {
    "stanford university",
    "massachusetts institute of technology",
    "mit",
    "harvard university",
    "university of california berkeley",
    "uc berkeley",
    "carnegie mellon university",
    "cmu",
    "university of oxford",
    "university of cambridge",
    "national university of singapore",
}
STRONG_UNIVERSITIES = {
    "northeastern university",
    "university of southern california",
    "usc",
    "new york university",
    "nyu",
    "university of texas at dallas",
    "ut dallas",
    "arizona state university",
    "university of waterloo",
    "university of toronto",
}


@dataclass
class ApplicantProfile:
    requested_loan_amount: float | None = None
    monthly_income: float | None = None
    existing_monthly_emi: float = 0.0
    credit_score: int | None = None
    age: int | None = None
    employment_type: str | None = None
    gre_score: int | None = None
    university_choice: str | None = None
    study_location: str | None = None
    collateral_available: bool = False
    citizenship: str | None = None


@dataclass
class ChatState:
    profile: ApplicantProfile = field(default_factory=ApplicantProfile)
    history: list[dict[str, str]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def parse_money_to_rupees(text: str) -> float | None:
    normalized = text.lower().replace(",", "")
    patterns = [
        (r"(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\s*(?:l|lac|lakh|lakhs)\b", 100_000),
        (r"(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\s*(?:cr|crore|crores)\b", 10_000_000),
        (r"(?:rs\.?|inr|₹)\s*(\d+(?:\.\d+)?)\b", 1),
    ]
    for pattern, multiplier in patterns:
        match = re.search(pattern, normalized)
        if match:
            return float(match.group(1)) * multiplier
    return None


def rupees_to_lakh(amount: float | None) -> str:
    if amount is None:
        return "unknown"
    return f"{amount / 100_000:.2f}L"


def update_profile_from_message(profile: ApplicantProfile, message: str) -> None:
    lower = message.lower()
    money = parse_money_to_rupees(message)

    if money and any(word in lower for word in ["loan", "borrow", "amount", "50l", "50 l"]):
        profile.requested_loan_amount = money
    elif money and any(word in lower for word in ["income", "salary", "earn", "monthly"]):
        profile.monthly_income = money
    elif money and any(word in lower for word in ["emi", "obligation", "debt"]):
        profile.existing_monthly_emi = money

    income_match = re.search(r"(?:income|salary|earn)\D{0,20}(\d+(?:\.\d+)?)\s*(?:k|thousand)\b", lower)
    if income_match:
        profile.monthly_income = float(income_match.group(1)) * 1_000

    emi_match = re.search(r"(?:emi|obligation|debt)\D{0,20}(\d+(?:\.\d+)?)\s*(?:k|thousand)\b", lower)
    if emi_match:
        profile.existing_monthly_emi = float(emi_match.group(1)) * 1_000

    score_match = re.search(r"(?:credit|cibil|score)\D{0,20}([3-9]\d{2})\b", lower)
    if score_match:
        profile.credit_score = int(score_match.group(1))

    age_match = re.search(r"(?:age|i am|i'm)\D{0,10}([1-9]\d)\b", lower)
    if age_match:
        profile.age = int(age_match.group(1))

    if "salaried" in lower:
        profile.employment_type = "salaried"
    elif "self employed" in lower or "self-employed" in lower or "business" in lower:
        profile.employment_type = "self-employed"

    gre_match = re.search(r"\bgre(?:\s*score)?\D{0,10}([2-3]\d{2})\b", lower)
    if gre_match:
        profile.gre_score = int(gre_match.group(1))

    university_match = re.search(r"(?:university|college|institute)\s*(?:choice|is|:|-)?\s*([a-z][a-z .&'-]{2,})", lower)
    if university_match:
        profile.university_choice = clean_entity(university_match.group(1))
    else:
        admitted_match = re.search(r"(?:admit|admission|offer)\s+(?:from|at|to)\s+([a-z][a-z .&'-]{2,})", lower)
        if admitted_match:
            profile.university_choice = clean_entity(admitted_match.group(1))

    for location in sorted(POONAWALLA_SUPPORTED_LOCATIONS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(location)}\b", lower):
            profile.study_location = location
            break

    if "without collateral" in lower or "no collateral" in lower or "unsecured" in lower:
        profile.collateral_available = False
    elif "collateral" in lower and any(word in lower for word in ["yes", "available", "have", "secured"]):
        profile.collateral_available = True

    if "oci" in lower:
        profile.citizenship = "OCI"
    elif "indian" in lower:
        profile.citizenship = "Indian"


def clean_entity(value: str) -> str:
    value = re.split(r",|\.|\band\b|\bfor\b|\bwith\b", value, maxsplit=1)[0]
    return re.sub(r"\s+", " ", value).strip().title()


def missing_fields(profile: ApplicantProfile) -> list[str]:
    missing: list[str] = []
    if profile.requested_loan_amount is None:
        missing.append("requested loan amount")
    if profile.monthly_income is None:
        missing.append("monthly income")
    if profile.credit_score is None:
        missing.append("credit score/CIBIL")
    if profile.age is None:
        missing.append("age")
    if profile.employment_type is None:
        missing.append("employment type")
    return missing


def missing_preapproval_fields(profile: ApplicantProfile) -> list[str]:
    missing: list[str] = []
    if profile.gre_score is None:
        missing.append("GRE score")
    if profile.university_choice is None:
        missing.append("university choice/admit")
    if profile.study_location is None:
        missing.append("study location/country")
    if profile.credit_score is None:
        missing.append("CIBIL score")
    if profile.age is None:
        missing.append("age")
    if profile.citizenship is None:
        missing.append("citizenship")
    return missing


def calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> float:
    monthly_rate = annual_rate / 12 / 100
    if monthly_rate == 0:
        return principal / tenure_months
    factor = math.pow(1 + monthly_rate, tenure_months)
    return principal * monthly_rate * factor / (factor - 1)


def evaluate_eligibility(profile: ApplicantProfile) -> dict[str, Any]:
    if profile.requested_loan_amount is None or profile.monthly_income is None:
        raise ValueError("Loan amount and monthly income are required for eligibility evaluation.")

    estimated_emi = calculate_emi(
        principal=profile.requested_loan_amount,
        annual_rate=DEFAULT_INTEREST_RATE,
        tenure_months=DEFAULT_TENURE_MONTHS,
    )
    total_obligations = estimated_emi + profile.existing_monthly_emi
    foi_ratio = total_obligations / profile.monthly_income

    checks = {
        "credit_score": {
            "passed": profile.credit_score is not None and profile.credit_score >= MIN_CREDIT_SCORE,
            "observed": profile.credit_score,
            "required": f">= {MIN_CREDIT_SCORE}",
        },
        "foir": {
            "passed": foi_ratio <= MAX_FOI_RATIO,
            "observed": round(foi_ratio, 3),
            "required": f"<= {MAX_FOI_RATIO}",
        },
        "age": {
            "passed": profile.age is not None and MIN_AGE <= profile.age <= MAX_AGE_AT_TENURE_END - (DEFAULT_TENURE_MONTHS // 12),
            "observed": profile.age,
            "required": f"{MIN_AGE}-{MAX_AGE_AT_TENURE_END - (DEFAULT_TENURE_MONTHS // 12)}",
        },
        "employment_type": {
            "passed": profile.employment_type in {"salaried", "self-employed"},
            "observed": profile.employment_type,
            "required": "salaried or self-employed",
        },
    }
    failed = [name for name, check in checks.items() if not check["passed"]]

    if not failed:
        decision = "eligible"
    elif failed == ["foir"] or failed == ["credit_score"]:
        decision = "conditional"
    else:
        decision = "not_eligible"

    return {
        "decision": decision,
        "requested_loan_amount": profile.requested_loan_amount,
        "requested_loan_amount_lakh": round(profile.requested_loan_amount / 100_000, 2),
        "estimated_emi": round(estimated_emi, 2),
        "monthly_income": profile.monthly_income,
        "existing_monthly_emi": profile.existing_monthly_emi,
        "foir": round(foi_ratio, 3),
        "policy": {
            "interest_rate": DEFAULT_INTEREST_RATE,
            "tenure_months": DEFAULT_TENURE_MONTHS,
            "max_foir": MAX_FOI_RATIO,
            "min_credit_score": MIN_CREDIT_SCORE,
        },
        "checks": checks,
        "failed_checks": failed,
        "note": "Demo rule engine only. A real lender would use verified documents, bureau data, risk scorecards, and bank policy.",
    }


def university_tier(university: str | None) -> str:
    if not university:
        return "unknown"
    normalized = university.lower()
    if normalized in SELECT_UNIVERSITIES:
        return "select"
    if normalized in STRONG_UNIVERSITIES:
        return "strong"
    return "standard"


def gre_multiplier(gre_score: int | None) -> float:
    if gre_score is None:
        return 0.0
    if gre_score >= 325:
        return 1.00
    if gre_score >= 315:
        return 0.85
    if gre_score >= 305:
        return 0.65
    if gre_score >= 295:
        return 0.40
    return 0.20


def university_multiplier(tier: str) -> float:
    return {"select": 1.00, "strong": 0.85, "standard": 0.60, "unknown": 0.0}[tier]


def process_poonawalla_preapproval(
    gre_score: int,
    university_choice: str,
    study_location: str,
    cibil_score: int,
    age: int,
    citizenship: str,
    collateral_available: bool = False,
) -> dict[str, Any]:
    """Demo pre-approval function based on public Poonawalla caps plus a sample GRE/university scorecard."""
    location = study_location.lower()
    tier = university_tier(university_choice)
    supported_location = location in POONAWALLA_SUPPORTED_LOCATIONS
    valid_citizenship = citizenship.lower() in {"indian", "oci"}
    valid_age = 16 <= age <= 50
    valid_cibil = cibil_score >= POONAWALLA_MIN_CIBIL

    if location == "india":
        product_cap = POONAWALLA_SECURED_DOMESTIC_CAP if collateral_available else POONAWALLA_MIN_LOAN
    elif collateral_available:
        product_cap = POONAWALLA_SECURED_ABROAD_CAP
    else:
        product_cap = POONAWALLA_UNSECURED_ABROAD_CAP

    score = 0.45 * gre_multiplier(gre_score) + 0.35 * university_multiplier(tier)
    score += 0.20 if valid_cibil else 0.05
    raw_limit = product_cap * score

    hard_failures = []
    if not supported_location:
        hard_failures.append("study_location_not_supported")
    if not valid_citizenship:
        hard_failures.append("citizenship_not_supported")
    if not valid_age:
        hard_failures.append("age_outside_policy")

    if hard_failures:
        preapproved_limit = 0
        decision = "not_preapproved"
    else:
        preapproved_limit = int(max(POONAWALLA_MIN_LOAN, min(product_cap, round(raw_limit / 100_000) * 100_000)))
        decision = "preapproved" if preapproved_limit >= 1_000_000 and valid_cibil else "conditional_preapproval"

    return {
        "decision": decision,
        "preapproved_limit": preapproved_limit,
        "preapproved_limit_lakh": round(preapproved_limit / 100_000, 2),
        "currency": "INR",
        "inputs": {
            "gre_score": gre_score,
            "university_choice": university_choice,
            "study_location": study_location,
            "cibil_score": cibil_score,
            "age": age,
            "citizenship": citizenship,
            "collateral_available": collateral_available,
        },
        "criteria": {
            "public_poonawalla_caps": {
                "unsecured_abroad": "Up to Rs 1 Cr",
                "secured_abroad": "Up to Rs 3 Cr",
                "secured_domestic": "Up to Rs 50 Lakh",
                "education_loan_range": "Rs 1 Lakh to Rs 3 Cr",
            },
            "public_poonawalla_eligibility": {
                "student_age": "16-50 years",
                "citizenship": "Indian or OCI",
                "supported_locations": sorted(POONAWALLA_SUPPORTED_LOCATIONS),
                "minimum_cibil_reference": "750+ used for this demo scorecard",
            },
            "demo_scorecard": {
                "gre_multiplier": gre_multiplier(gre_score),
                "university_tier": tier,
                "university_multiplier": university_multiplier(tier),
                "cibil_passed": valid_cibil,
            },
        },
        "hard_failures": hard_failures,
        "disclaimer": (
            "This is a demo function call, not an official Poonawalla Fincorp sanction. "
            "The public site gives broad caps and eligibility; exact GRE/university underwriting grids are lender policy."
        ),
    }


def ask_for_missing_fields(missing: list[str]) -> str:
    labels = ", ".join(missing)
    return (
        f"I can check that. Please share your {labels}. "
        "For example: monthly income 1.8L, existing EMI 20k, CIBIL 760, age 29, salaried."
    )


def ask_for_missing_preapproval_fields(missing: list[str]) -> str:
    labels = ", ".join(missing)
    return (
        f"I can call the Poonawalla pre-approval function once I have your {labels}. "
        "For example: GRE 322, admit from Northeastern University, USA, CIBIL 780, age 24, Indian, no collateral."
    )


def local_response(result: dict[str, Any]) -> str:
    failed = ", ".join(result["failed_checks"]) or "none"
    return (
        f"Eligibility result: {result['decision'].replace('_', ' ').title()}.\n"
        f"Requested amount: {rupees_to_lakh(result['requested_loan_amount'])}.\n"
        f"Estimated EMI: Rs {result['estimated_emi']:,.0f} for "
        f"{result['policy']['tenure_months']} months at {result['policy']['interest_rate']}% p.a.\n"
        f"FOIR: {result['foir']:.1%}; failed checks: {failed}.\n"
        f"{result['note']}"
    )


def preapproval_response(tool_call: dict[str, Any]) -> str:
    result = tool_call["result"]
    return (
        "Function call executed:\n"
        f"{tool_call['name']}({json.dumps(tool_call['arguments'])})\n\n"
        "Function result:\n"
        f"{json.dumps(result, indent=2)}\n\n"
        f"Pre-Approved Limit: Rs {result['preapproved_limit']:,.0f} "
        f"({result['preapproved_limit_lakh']:.2f}L). Decision: {result['decision'].replace('_', ' ').title()}."
    )


def call_pfl_preapproval_api(arguments: dict[str, Any]) -> dict[str, Any] | None:
    base_url = os.environ.get("PFL_MOCK_API_URL")
    if not base_url:
        return None

    url = f"{base_url.rstrip('/')}/v1/preapproval"
    payload = json.dumps(arguments).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {
            "decision": "api_error",
            "preapproved_limit": 0,
            "preapproved_limit_lakh": 0,
            "currency": "INR",
            "error": str(exc),
            "disclaimer": "Mock PFL API call failed. Falling back should be handled by the caller if desired.",
        }


def llm_response(result: dict[str, Any], user_message: str, model: str) -> str | None:
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except ImportError:
        return None

    llm = ChatOpenAI(model=model, temperature=0)
    messages = [
        (
            "system",
            "You are a concise loan eligibility assistant. Explain the rule-engine result clearly. "
            "Do not guarantee approval. Mention that final approval requires lender verification.",
        ),
        (
            "human",
            "User request:\n"
            f"{user_message}\n\n"
            "Eligibility engine JSON:\n"
            f"{json.dumps(result, indent=2)}",
        ),
    ]
    response = llm.invoke(messages)
    return getattr(response, "content", None) or getattr(response, "text", None)


def handle_message(state: ChatState, message: str, model: str) -> str:
    state.history.append({"role": "user", "content": message})
    update_profile_from_message(state.profile, message)

    if wants_poonawalla_preapproval(message, state.profile):
        missing = missing_preapproval_fields(state.profile)
        if missing:
            response = ask_for_missing_preapproval_fields(missing)
            state.history.append({"role": "assistant", "content": response})
            return response

        arguments = {
            "gre_score": state.profile.gre_score,
            "university_choice": state.profile.university_choice,
            "study_location": state.profile.study_location,
            "cibil_score": state.profile.credit_score,
            "age": state.profile.age,
            "citizenship": state.profile.citizenship,
            "collateral_available": state.profile.collateral_available,
        }
        api_result = call_pfl_preapproval_api(arguments)
        result = api_result or process_poonawalla_preapproval(**arguments)
        tool_call = {
            "name": "POST /v1/preapproval" if api_result else "process_poonawalla_preapproval",
            "arguments": arguments,
            "result": result,
        }
        state.tool_calls.append(tool_call)
        response = preapproval_response(tool_call)
        state.history.append({"role": "assistant", "content": response})
        return response

    missing = missing_fields(state.profile)
    if missing:
        response = ask_for_missing_fields(missing)
        state.history.append({"role": "assistant", "content": response})
        return response

    result = evaluate_eligibility(state.profile)
    response = llm_response(result, message, model) or local_response(result)
    state.history.append({"role": "assistant", "content": response})
    return response


def wants_poonawalla_preapproval(message: str, profile: ApplicantProfile) -> bool:
    lower = message.lower()
    trigger_words = ["poonawalla", "pre-approved", "preapproved", "pre approved", "gre", "university"]
    return any(word in lower for word in trigger_words) or (
        profile.gre_score is not None and profile.university_choice is not None
    )


def run_chat(model: str) -> None:
    state = ChatState()
    print("Agentic Loan Orchestrator")
    print('Try: "Check my eligibility for a 50L loan."')
    print('Or: "GRE 322, admit from Northeastern University, USA, CIBIL 780, age 24, Indian, no collateral."')
    print('Type "exit" to quit.')
    while True:
        message = input("\nYou: ").strip()
        if message.lower() in {"exit", "quit"}:
            print("Assistant: Goodbye.")
            return
        if not message:
            continue
        print(f"\nAssistant: {handle_message(state, message, model)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat interface for agentic loan eligibility checks.")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model used through LangChain when OPENAI_API_KEY is set.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_chat(args.model)
