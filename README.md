# agentX
Section 1: Executive Summary

agentX is a high-speed customer acquisition engine designed to plug into PFL's AI-First stack.

It combines OCR-based document intelligence, Agentic DQI checks, loan eligibility orchestration, and mock LOS/RegIntel integration patterns to accelerate education-loan journeys from document upload to pre-approved limit discovery. The system is built to demonstrate how AI agents can collect applicant data, validate document quality, call lending-policy functions, and return structured decisions with minimal operational friction.

Section 2: The Tech Stack
AWS: S3 for secure document intake and storage, Lambda for event-driven OCR, DQI, and eligibility workflows.
Python: Core OCR, DQI, loan-policy checks, mock API, and orchestration logic.
React: Customer-facing chat and upload interface for guided loan journeys.
LangChain: Agentic orchestration layer for chat-driven workflows, function calls, and OpenAI-backed response generation.
Section 3: The "Risk-First" Layer
agentX is designed with a Risk-First layer that validates applicant identity and document quality before pushing a case toward approval.

The current code demonstrates this through OCR and DQI validation for PAN cards, marksheets, and university offer letters. The same pattern extends to KYC checks:

Aadhaar validation: Extract Aadhaar details, verify format, compare applicant name/date of birth, and flag missing or low-confidence fields.
PAN validation: Extract PAN number, name, father's name, and date of birth; validate PAN format and compare the PAN name against the University Offer Letter.
Face Match: Compare the applicant selfie with document photo evidence, returning a match score and anomaly flag when confidence is below policy threshold.
These checks are intended to feed a structured DQI and RegIntel response so only clean, policy-compliant applications move forward. By surfacing anomalies automatically, agentX supports Zero-Manual Intervention for straight-through cases while routing only exceptions for review.

```markdown
# OCR & DQI Engine (Python/EasyOCR)

This project extracts key fields from uploaded documents such as PAN cards and marksheets, then calculates a Data Quality Index (DQI) score.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

PDF input requires Poppler to be installed and available on `PATH`.

## Usage

```powershell
python ocr_dqi_engine.py path\to\pan.jpg --doc-type pan
python ocr_dqi_engine.py path\to\marksheet.pdf --doc-type marksheet --output result.json
python ocr_dqi_engine.py path\to\document.png --doc-type auto
python ocr_dqi_engine.py path\to\pan.jpg --doc-type pan --offer-letter path\to\offer-letter.pdf
```

## Extracted Fields

PAN cards:

- PAN number
- Name
- Father's name
- Date of birth

Marksheets:

- Student name
- Roll/seat/registration number
- Year
- Percentage
- Total marks
- Subject rows when visible as `Subject 78/100`

## DQI

The DQI score combines:

- Completeness of required fields
- Format/range validations
- Average OCR confidence

The output JSON includes the final score, grade, missing fields, failed validations, raw OCR text, bounding boxes, and confidence values.

## Agentic DQI

For PAN verification against a University Offer Letter, pass `--offer-letter`.

The engine extracts the PAN name and the offer-letter applicant/student name, compares them with fuzzy matching, and adds an `agentic_dqi` section. If the names do not match the configured threshold, the output includes a `"Data Quality Anomaly"` with the PAN name, offer-letter name, similarity score, and rule that failed.

```powershell
python ocr_dqi_engine.py uploads\pan.jpg --doc-type pan --offer-letter uploads\offer-letter.pdf --output result.json
```

## Agentic Orchestrator

`agentic_orchestrator.py` is a separate LangChain/OpenAI-ready chat script for loan eligibility conversations.

```powershell
python agentic_orchestrator.py
```

Example prompt:

```text
Check my eligibility for a 50L loan.
```

The orchestrator asks for missing inputs such as monthly income, current EMI, CIBIL score, age, and employment type. It then runs a deterministic eligibility tool and optionally uses LangChain `ChatOpenAI` to summarize the result when `OPENAI_API_KEY` is set.

For education-loan pre-approval, the orchestrator now calls a function instead of only replying with text:

```text
GRE 322, admit from Northeastern University, USA, CIBIL 780, age 24, Indian, no collateral.
```

This executes `process_poonawalla_preapproval(...)`, compares GRE score and university tier against a demo scorecard, applies public Poonawalla Fincorp education-loan caps, and returns a pre-approved limit. The public caps and eligibility are broad lender information; the GRE/university scoring grid is intentionally marked as a replaceable demo policy.

### Mock PFL API

The `mock_api` folder contains a local mock PFL API with `POST /v1/preapproval`.

```powershell
python mock_api/server.py --port 8080
$env:PFL_MOCK_API_URL = "http://127.0.0.1:8080"
python agentic_orchestrator.py
```



With `PFL_MOCK_API_URL` set, the orchestrator calls the mock API instead of the in-process function.
```
