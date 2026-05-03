#!/usr/bin/env python3
"""
AgentX Single-File App

Run:
    python agentx_single_app.py --port 8090

Open:
    http://127.0.0.1:8090

This one file includes:
    - Browser chat interface
    - Agentic loan orchestrator
    - Mock PFL pre-approval API
    - LOS + RegIntel mock integration response
    - OCR/DQI upload endpoint with optional EasyOCR support
"""

from __future__ import annotations

import argparse
import cgi
import difflib
import json
import math
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


APP_NAME = "AgentX"
HOST = "127.0.0.1"
DEFAULT_PORT = 8090
DEFAULT_INTEREST_RATE = 10.5
DEFAULT_TENURE_MONTHS = 60
MAX_FOIR = 0.50
MIN_CREDIT_SCORE = 700
PFL_MIN_CIBIL = 750
PFL_UNSECURED_ABROAD_CAP = 10_000_000
PFL_SECURED_ABROAD_CAP = 30_000_000
PFL_SECURED_DOMESTIC_CAP = 5_000_000
PFL_MIN_LOAN = 100_000
PAN_PATTERN = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
DATE_PATTERN = re.compile(r"\b([0-3]?\d[/-][01]?\d[/-](?:19|20)?\d{2})\b")

SUPPORTED_LOCATIONS = {
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
class AppState:
    profile: ApplicantProfile = field(default_factory=ApplicantProfile)
    messages: list[dict[str, str]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


STATE = AppState()


def parse_money(text: str) -> float | None:
    text = text.lower().replace(",", "")
    patterns = [
        (r"(?:rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:l|lac|lakh|lakhs)\b", 100_000),
        (r"(?:rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:cr|crore|crores)\b", 10_000_000),
        (r"(?:rs\.?|inr)\s*(\d+(?:\.\d+)?)\b", 1),
    ]
    for pattern, multiplier in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1)) * multiplier
    return None


def clean_entity(value: str) -> str:
    value = re.split(r",|\.|\band\b|\bfor\b|\bwith\b", value, maxsplit=1)[0]
    return re.sub(r"\s+", " ", value).strip().title()


def rupees_lakh(amount: float | None) -> str:
    return "unknown" if amount is None else f"{amount / 100_000:.2f}L"


def update_profile(profile: ApplicantProfile, message: str) -> None:
    lower = message.lower()
    money = parse_money(message)
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

    gre_match = re.search(r"\bgre(?:\s*score)?\D{0,10}([2-3]\d{2})\b", lower)
    if gre_match:
        profile.gre_score = int(gre_match.group(1))

    university_match = re.search(r"(?:university|college|institute)\s*(?:choice|is|:|-)?\s*([a-z][a-z .&'-]{2,})", lower)
    admit_match = re.search(r"(?:admit|admission|offer)\s+(?:from|at|to)\s+([a-z][a-z .&'-]{2,})", lower)
    if university_match:
        profile.university_choice = clean_entity(university_match.group(1))
    elif admit_match:
        profile.university_choice = clean_entity(admit_match.group(1))

    for location in sorted(SUPPORTED_LOCATIONS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(location)}\b", lower):
            profile.study_location = location
            break

    if "salaried" in lower:
        profile.employment_type = "salaried"
    elif "self employed" in lower or "self-employed" in lower or "business" in lower:
        profile.employment_type = "self-employed"

    if "without collateral" in lower or "no collateral" in lower or "unsecured" in lower:
        profile.collateral_available = False
    elif "collateral" in lower and any(word in lower for word in ["yes", "available", "have", "secured"]):
        profile.collateral_available = True

    if "oci" in lower:
        profile.citizenship = "OCI"
    elif "indian" in lower:
        profile.citizenship = "Indian"


def calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> float:
    monthly_rate = annual_rate / 12 / 100
    factor = math.pow(1 + monthly_rate, tenure_months)
    return principal * monthly_rate * factor / (factor - 1)


def university_tier(university: str | None) -> str:
    normalized = (university or "").lower()
    if normalized in SELECT_UNIVERSITIES:
        return "select"
    if normalized in STRONG_UNIVERSITIES:
        return "strong"
    return "standard" if normalized else "unknown"


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
    location = study_location.lower()
    tier = university_tier(university_choice)
    supported_location = location in SUPPORTED_LOCATIONS
    valid_citizenship = citizenship.lower() in {"indian", "oci"}
    valid_age = 16 <= age <= 50
    valid_cibil = cibil_score >= PFL_MIN_CIBIL

    if location == "india":
        product_cap = PFL_SECURED_DOMESTIC_CAP if collateral_available else PFL_MIN_LOAN
    elif collateral_available:
        product_cap = PFL_SECURED_ABROAD_CAP
    else:
        product_cap = PFL_UNSECURED_ABROAD_CAP

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
        limit = 0
        decision = "not_preapproved"
    else:
        limit = int(max(PFL_MIN_LOAN, min(product_cap, round(raw_limit / 100_000) * 100_000)))
        decision = "preapproved" if limit >= 1_000_000 and valid_cibil else "conditional_preapproval"

    return {
        "provider": "Poonawalla Fincorp Limited Mock",
        "decision": decision,
        "preapproved_limit": limit,
        "preapproved_limit_lakh": round(limit / 100_000, 2),
        "currency": "INR",
        "criteria": {
            "public_caps": {
                "unsecured_abroad": "Up to Rs 1 Cr",
                "secured_abroad": "Up to Rs 3 Cr",
                "secured_domestic": "Up to Rs 50 Lakh",
            },
            "demo_scorecard": {
                "gre_multiplier": gre_multiplier(gre_score),
                "university_tier": tier,
                "university_multiplier": university_multiplier(tier),
                "cibil_passed": valid_cibil,
            },
        },
        "inputs": {
            "gre_score": gre_score,
            "university_choice": university_choice,
            "study_location": study_location,
            "cibil_score": cibil_score,
            "age": age,
            "citizenship": citizenship,
            "collateral_available": collateral_available,
        },
        "hard_failures": hard_failures,
        "disclaimer": "Mock response only. Final sanction requires lender verification.",
    }


def los_regintel_response(preapproval: dict[str, Any]) -> dict[str, Any]:
    passed = preapproval["decision"] in {"preapproved", "conditional_preapproval"}
    return {
        "integration": {
            "provider": "Poonawalla Fincorp Limited",
            "environment": "mock",
            "systems": ["LOS", "RegIntel"],
            "request_id": f"pfl_mock_{uuid.uuid4().hex[:10]}",
        },
        "los_response": {
            "application_id": f"LOS-EDU-{uuid.uuid4().hex[:6].upper()}",
            "product": "Education Loan",
            "status": "PRE_APPROVED" if passed else "REFER",
            "preapproved_limit": preapproval["preapproved_limit"],
            "currency": "INR",
            "conditions": [
                "Final sanction subject to document verification",
                "Institute admit letter required",
                "Applicant and co-applicant KYC required",
            ],
        },
        "regintel_response": {
            "compliance_status": "PASS" if not preapproval["hard_failures"] else "REFER",
            "checks": {
                "kyc_required": "PENDING_UPLOAD",
                "pan_validation": "PENDING",
                "age_policy": "PASS" if "age_outside_policy" not in preapproval["hard_failures"] else "FAIL",
                "citizenship_policy": "PASS" if "citizenship_not_supported" not in preapproval["hard_failures"] else "FAIL",
                "study_location_policy": "PASS" if "study_location_not_supported" not in preapproval["hard_failures"] else "FAIL",
                "sanctions_screening": "NO_MATCH",
            },
        },
        "agentic_decision": {
            "decision": "PRE_APPROVED_WITH_CONDITIONS" if passed else "MANUAL_REVIEW",
            "next_action": "Collect KYC, admit letter, GRE proof, co-applicant details, and financial documents.",
        },
    }


def missing_preapproval_fields(profile: ApplicantProfile) -> list[str]:
    missing = []
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


def missing_loan_fields(profile: ApplicantProfile) -> list[str]:
    missing = []
    if profile.requested_loan_amount is None:
        missing.append("requested loan amount")
    if profile.monthly_income is None:
        missing.append("monthly income")
    if profile.credit_score is None:
        missing.append("CIBIL score")
    if profile.age is None:
        missing.append("age")
    if profile.employment_type is None:
        missing.append("employment type")
    return missing


def handle_chat(message: str) -> dict[str, Any]:
    STATE.messages.append({"role": "user", "content": message})
    update_profile(STATE.profile, message)

    lower = message.lower()
    wants_preapproval = any(word in lower for word in ["poonawalla", "preapproved", "pre-approved", "pre approved", "gre", "university"])
    if wants_preapproval or (STATE.profile.gre_score and STATE.profile.university_choice):
        missing = missing_preapproval_fields(STATE.profile)
        if missing:
            reply = "I can call the PFL pre-approval function once I have your " + ", ".join(missing) + "."
            STATE.messages.append({"role": "assistant", "content": reply})
            return {"reply": reply, "profile": STATE.profile.__dict__}

        args = {
            "gre_score": STATE.profile.gre_score,
            "university_choice": STATE.profile.university_choice,
            "study_location": STATE.profile.study_location,
            "cibil_score": STATE.profile.credit_score,
            "age": STATE.profile.age,
            "citizenship": STATE.profile.citizenship,
            "collateral_available": STATE.profile.collateral_available,
        }
        result = process_poonawalla_preapproval(**args)
        integration = los_regintel_response(result)
        tool_call = {"name": "POST /v1/preapproval", "arguments": args, "result": result}
        STATE.tool_calls.append(tool_call)
        reply = f"Function call executed: POST /v1/preapproval. Pre-Approved Limit: Rs {result['preapproved_limit']:,.0f} ({result['preapproved_limit_lakh']:.2f}L)."
        STATE.messages.append({"role": "assistant", "content": reply})
        return {"reply": reply, "tool_call": tool_call, "los_regintel": integration, "profile": STATE.profile.__dict__}

    missing = missing_loan_fields(STATE.profile)
    if missing:
        reply = "I can check that. Please share your " + ", ".join(missing) + "."
        STATE.messages.append({"role": "assistant", "content": reply})
        return {"reply": reply, "profile": STATE.profile.__dict__}

    estimated_emi = calculate_emi(STATE.profile.requested_loan_amount, DEFAULT_INTEREST_RATE, DEFAULT_TENURE_MONTHS)
    foir = (estimated_emi + STATE.profile.existing_monthly_emi) / STATE.profile.monthly_income
    failed = []
    if STATE.profile.credit_score < MIN_CREDIT_SCORE:
        failed.append("credit_score")
    if foir > MAX_FOIR:
        failed.append("foir")
    decision = "eligible" if not failed else "conditional"
    result = {
        "decision": decision,
        "requested_loan_amount": STATE.profile.requested_loan_amount,
        "estimated_emi": round(estimated_emi, 2),
        "foir": round(foir, 3),
        "failed_checks": failed,
    }
    reply = f"Eligibility result: {decision.title()}. Requested amount: {rupees_lakh(STATE.profile.requested_loan_amount)}. Estimated EMI: Rs {estimated_emi:,.0f}. FOIR: {foir:.1%}."
    STATE.messages.append({"role": "assistant", "content": reply})
    return {"reply": reply, "loan_result": result, "profile": STATE.profile.__dict__}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("|", "I")).strip()


def extract_text_with_easyocr(path: Path) -> list[dict[str, Any]]:
    try:
        import easyocr  # type: ignore
    except ImportError:
        return [{"text": "EasyOCR not installed. Install requirements.txt to run live OCR.", "confidence": 0.0, "box": []}]

    reader = easyocr.Reader(["en"], gpu=False)
    results = reader.readtext(str(path), detail=1, paragraph=False)
    lines = []
    for box, text, confidence in results:
        cleaned = normalize_text(text)
        if cleaned:
            lines.append({"text": cleaned, "confidence": float(confidence), "box": box})
    return lines


def build_ocr_dqi(lines: list[dict[str, Any]], doc_type: str = "auto") -> dict[str, Any]:
    raw_text = "\n".join(line["text"] for line in lines)
    upper = raw_text.upper()
    actual_type = doc_type
    if actual_type == "auto":
        actual_type = "pan" if PAN_PATTERN.search(upper) else "generic"

    fields: dict[str, Any] = {}
    if actual_type == "pan":
        names = [
            line["text"].upper()
            for line in lines
            if re.fullmatch(r"[A-Z][A-Z .'-]{2,}", line["text"].upper())
            and line["text"].upper() not in {"INCOME TAX DEPARTMENT", "GOVERNMENT OF INDIA", "GOVT OF INDIA"}
        ]
        fields = {
            "pan_number": first_match(PAN_PATTERN, upper),
            "name": names[0] if names else None,
            "father_name": names[1] if len(names) > 1 else None,
            "date_of_birth": first_match(DATE_PATTERN, upper),
        }
        required = ["pan_number", "name", "father_name", "date_of_birth"]
    else:
        fields = {"pan_number": first_match(PAN_PATTERN, upper), "date": first_match(DATE_PATTERN, upper)}
        required = []

    missing = [field for field in required if not fields.get(field)]
    confidence = sum(line.get("confidence", 0.0) for line in lines) / len(lines) if lines else 0.0
    completeness = (len(required) - len(missing)) / len(required) if required else 1.0
    score = round((0.70 * completeness + 0.30 * confidence) * 100, 2)
    return {
        "document_type": actual_type,
        "fields": fields,
        "dqi": {
            "score": score,
            "grade": "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D",
            "missing_fields": missing,
            "ocr_confidence": round(confidence, 3),
        },
        "ocr": lines,
    }


def first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def name_similarity(left: str | None, right: str | None) -> float:
    if not left or not right:
        return 0.0
    left = re.sub(r"[^A-Z ]", " ", left.upper())
    right = re.sub(r"[^A-Z ]", " ", right.upper())
    return round(difflib.SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio(), 3)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AgentX</title>
  <style>
    :root { color-scheme: light; --ink:#17202a; --muted:#657080; --line:#d8dde5; --panel:#f7f8fb; --accent:#0b6b5f; --warn:#9a3412; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Arial, sans-serif; color:var(--ink); background:#ffffff; }
    header { padding:18px 24px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; align-items:center; }
    h1 { margin:0; font-size:22px; letter-spacing:0; }
    main { display:grid; grid-template-columns: 1.1fr 0.9fr; min-height: calc(100vh - 65px); }
    section { padding:22px; border-right:1px solid var(--line); }
    aside { padding:22px; background:var(--panel); overflow:auto; }
    h2 { margin:0 0 12px; font-size:17px; }
    .chat { height:42vh; overflow:auto; border:1px solid var(--line); padding:12px; background:#fff; }
    .msg { padding:9px 10px; margin:8px 0; border-radius:6px; max-width:92%; white-space:pre-wrap; line-height:1.35; }
    .user { margin-left:auto; background:#e8f2ef; }
    .assistant { background:#f0f2f5; }
    .row { display:flex; gap:8px; margin-top:10px; }
    input[type=text] { flex:1; padding:11px; border:1px solid var(--line); border-radius:6px; font-size:14px; }
    button { border:0; background:var(--accent); color:white; padding:10px 14px; border-radius:6px; cursor:pointer; font-weight:700; }
    button.secondary { background:#334155; }
    button.warn { background:var(--warn); }
    pre { background:#101828; color:#e5edf7; padding:14px; border-radius:6px; overflow:auto; font-size:12px; line-height:1.45; }
    .tools { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }
    .upload { padding:14px; border:1px dashed var(--line); background:#fff; border-radius:6px; margin-top:18px; }
    .small { color:var(--muted); font-size:13px; line-height:1.4; }
    @media (max-width: 900px) { main { grid-template-columns:1fr; } section { border-right:0; border-bottom:1px solid var(--line); } }
  </style>
</head>
<body>
  <header>
    <h1>AgentX</h1>
    <div class="small">Single-file AI acquisition demo</div>
  </header>
  <main>
    <section>
      <h2>Agentic Orchestrator</h2>
      <div id="chat" class="chat"></div>
      <div class="row">
        <input id="message" type="text" value="GRE 322, admit from Northeastern University, USA, CIBIL 780, age 24, Indian, no collateral." />
        <button onclick="sendChat()">Send</button>
      </div>
      <div class="tools">
        <button class="secondary" onclick="quick('Check my eligibility for a 50L loan.')">50L Loan</button>
        <button class="secondary" onclick="quick('monthly income 1.8L, existing EMI 20k, CIBIL 760, age 29, salaried')">Add Income</button>
        <button class="warn" onclick="resetState()">Reset</button>
      </div>
      <div class="upload">
        <h2>OCR/DQI Upload</h2>
        <p class="small">Upload a PAN image to run optional EasyOCR. If EasyOCR is not installed, the endpoint returns a clear placeholder response.</p>
        <input id="doc" type="file" />
        <button onclick="uploadDoc()">Run OCR/DQI</button>
      </div>
    </section>
    <aside>
      <h2>Latest JSON</h2>
      <pre id="json">{}</pre>
    </aside>
  </main>
  <script>
    const chat = document.getElementById('chat');
    const out = document.getElementById('json');
    function add(role, text) {
      const div = document.createElement('div');
      div.className = 'msg ' + role;
      div.textContent = text;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }
    function quick(text) {
      document.getElementById('message').value = text;
      sendChat();
    }
    async function sendChat() {
      const input = document.getElementById('message');
      const message = input.value.trim();
      if (!message) return;
      add('user', message);
      const res = await fetch('/api/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({message}) });
      const data = await res.json();
      add('assistant', data.reply || JSON.stringify(data));
      out.textContent = JSON.stringify(data, null, 2);
    }
    async function uploadDoc() {
      const file = document.getElementById('doc').files[0];
      if (!file) return;
      const form = new FormData();
      form.append('document', file);
      form.append('doc_type', 'auto');
      const res = await fetch('/api/ocr-dqi', { method:'POST', body: form });
      const data = await res.json();
      out.textContent = JSON.stringify(data, null, 2);
      add('assistant', 'OCR/DQI completed. Score: ' + (data.dqi ? data.dqi.score : 'n/a'));
    }
    async function resetState() {
      const res = await fetch('/api/reset', { method:'POST' });
      const data = await res.json();
      chat.innerHTML = '';
      out.textContent = JSON.stringify(data, null, 2);
    }
    add('assistant', 'Ask for a 50L loan check, or provide GRE + university details to call the mock PFL pre-approval API.');
  </script>
</body>
</html>"""


class AgentXHandler(BaseHTTPRequestHandler):
    server_version = "AgentX/1.0"

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self.send_html(HTML)
        elif self.path == "/health":
            self.send_json(200, {"status": "ok", "service": APP_NAME})
        else:
            self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path == "/api/chat":
            self.send_json(200, handle_chat(self.read_json().get("message", "")))
        elif self.path == "/v1/preapproval":
            payload = self.read_json()
            required = ["gre_score", "university_choice", "study_location", "cibil_score", "age", "citizenship"]
            missing = [field for field in required if field not in payload]
            if missing:
                self.send_json(400, {"error": "missing_fields", "fields": missing})
                return
            result = process_poonawalla_preapproval(
                gre_score=int(payload["gre_score"]),
                university_choice=str(payload["university_choice"]),
                study_location=str(payload["study_location"]),
                cibil_score=int(payload["cibil_score"]),
                age=int(payload["age"]),
                citizenship=str(payload["citizenship"]),
                collateral_available=bool(payload.get("collateral_available", False)),
            )
            self.send_json(200, result)
        elif self.path == "/api/los-regintel":
            self.send_json(200, los_regintel_response(self.read_json()))
        elif self.path == "/api/reset":
            global STATE
            STATE = AppState()
            self.send_json(200, {"status": "reset"})
        elif self.path == "/api/ocr-dqi":
            self.handle_ocr_upload()
        else:
            self.send_json(404, {"error": "not_found"})

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        return json.loads(raw or "{}")

    def handle_ocr_upload(self) -> None:
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
        if "document" not in form:
            self.send_json(400, {"error": "document file is required"})
            return
        item = form["document"]
        doc_type = form.getvalue("doc_type", "auto")
        filename = Path(item.filename or "upload.bin").name
        with tempfile.TemporaryDirectory(prefix="agentx_upload_") as tmp:
            path = Path(tmp) / filename
            path.write_bytes(item.file.read())
            lines = extract_text_with_easyocr(path)
            self.send_json(200, build_ocr_dqi(lines, doc_type))

    def send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        print("agentx:", fmt % args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AgentX single-file app.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AgentXHandler)
    print(f"AgentX running at http://{args.host}:{args.port}")
    print("Routes: /, /api/chat, /v1/preapproval, /api/ocr-dqi, /api/los-regintel")
    server.serve_forever()


if __name__ == "__main__":
    main()
