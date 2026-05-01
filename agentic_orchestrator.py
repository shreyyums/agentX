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
