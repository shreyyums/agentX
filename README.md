
# agentX — AI-First Customer Acquisition Engine for Education Loans

> Built for the Poonawalla Fincorp Hackathon · Mock Demo 

agentX is a high-speed loan origination intelligence layer that combines agentic chat, OCR-based document intelligence, risk-first DQI validation, and mock LOS/RegIntel integration — all designed to plug into PFL's AI-First stack and compress the education-loan journey from document upload to pre-approved limit discovery.

---

## What agentX Does

Traditional loan journeys are slow, manual, and opaque. agentX inverts that with an agent that:

- **Talks** — a browser-based chat interface that guides applicants through the loan flow naturally
- **Reads** — OCR-powered document extraction for PAN cards, marksheets, and university offer letters
- **Validates** — a Risk-First DQI (Data Quality Integrity) layer that catches anomalies before they enter the pipeline
- **Decides** — deterministic loan eligibility checks and a mock PFL pre-approval engine
- **Hands off** — structured JSON payloads simulating LOS and RegIntel integration for downstream consumption

---

## Quickstart

```powershell
python agentx_single_app.py --port 8092
```

Then open: `http://127.0.0.1:8092`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Cloud & Storage | AWS S3 (document intake), Lambda (event-driven workflows) |
| Backend | Python — OCR, DQI, eligibility, mock API, orchestration |
| Frontend | React — chat UI and document upload interface |
| Agentic Layer | LangChain + OpenAI — chat-driven workflows and function calls |

---

## Core Features

###  Autonomous Intent Detection

agentX detects when a user is describing a university, GRE score, study destination, or pre-approval scenario — without requiring them to navigate menus or select a workflow.

**Example prompt:**
```
GRE 322, admit from Northeastern University, USA, CIBIL 780, age 24, Indian, no collateral.
```

This single message triggers the full pre-approval workflow: intent extraction → input parsing → mock PFL API call → pre-approved limit → LOS/RegIntel JSON generation.

---

###  OCR & DQI Engine

Powered by EasyOCR (Python). Extracts structured fields from uploaded documents and scores them with a Data Quality Index.

**Supported documents and extracted fields:**

| Document | Extracted Fields |
|---|---|
| PAN Card | PAN number, name, father's name, date of birth |
| Marksheet | Student name, roll/seat/registration number, year, percentage, total marks, subject rows |
| University Offer Letter | Applicant name (for cross-document validation) |

**DQI Score Components:**
- Completeness of required fields
- Format and range validations (e.g. `AAAAA9999A` PAN pattern)
- Average OCR confidence

**Output includes:** DQI score, grade, missing fields, failed validations, raw OCR text, bounding boxes, and confidence values.

```powershell
# Basic usage
python ocr_dqi_engine.py path\to\pan.jpg --doc-type pan

# With marksheet output
python ocr_dqi_engine.py path\to\marksheet.pdf --doc-type marksheet --output result.json

# Auto-detect document type
python ocr_dqi_engine.py path\to\document.png --doc-type auto

# Cross-document PAN vs Offer Letter validation
python ocr_dqi_engine.py path\to\pan.jpg --doc-type pan --offer-letter path\to\offer-letter.pdf
```

---

###  Risk-First Layer

agentX validates applicant identity and document quality *before* any case is pushed toward approval. Only clean, policy-compliant applications proceed — exceptions are routed for manual review.

**KYC Checks:**

- **Aadhaar Validation** — extracts Aadhaar details, verifies format, compares name/DOB, flags missing or low-confidence fields
- **PAN Validation** — extracts PAN number, name, father's name, and DOB; validates format; cross-checks PAN name against University Offer Letter using fuzzy matching
- **Face Match** — compares applicant selfie against document photo evidence; returns match score and anomaly flag when confidence falls below policy threshold

**Agentic DQI (Cross-Document):**

When `--offer-letter` is passed, the engine compares the extracted PAN name against the offer letter applicant name with fuzzy matching. If similarity falls below the configured threshold, an `agentic_dqi` anomaly block is added to the output — surfacing discrepancies automatically instead of silently passing bad data forward.

This supports **Zero-Manual Intervention** for straight-through cases while routing only exceptions for human review.

---

###  Agentic Orchestrator

`agentic_orchestrator.py` is a LangChain/OpenAI-powered chat agent for loan eligibility conversations.

```powershell
python agentic_orchestrator.py
```

**Example prompt:**
```
Check my eligibility for a 50L loan.
```

The orchestrator asks for missing inputs (monthly income, current EMI, CIBIL score, age, employment type), runs a deterministic eligibility tool, and optionally uses LangChain `ChatOpenAI` to summarize the result when `OPENAI_API_KEY` is set.

**Education Loan Pre-Approval flow:**
```
GRE 322, admit from Northeastern University, USA, CIBIL 780, age 24, Indian, no collateral.
```
This executes `process_poonawalla_preapproval(...)`, which compares GRE score and university tier against a demo scorecard, applies public Poonawalla Fincorp education-loan caps, and returns a pre-approved limit.

---

###  Mock PFL API & Bank-Grade Handshake

The `mock_api` folder exposes a local mock PFL API at `POST /v1/preapproval`.

```powershell
python mock_api/server.py --port 8080
$env:PFL_MOCK_API_URL = "http://127.0.0.1:8080"
python agentic_orchestrator.py
```

When `PFL_MOCK_API_URL` is set, the orchestrator routes calls to the mock API instead of the in-process function.

When the pre-approval workflow runs, agentX generates a structured JSON handshake simulating a real banking integration:

```json
{
  "systems": ["LOS", "RegIntel"],
  "environment": "mock",
  "product": "Education Loan"
}
```

The handshake includes:
- Mock PFL pre-approval response
- Mock Loan Origination System (LOS) application response
- Mock RegIntel compliance response
- Agentic decision block explaining the next recommended action

---

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> PDF input requires **Poppler** to be installed and available on PATH.

---

## End-to-End Example Flow

1. User inputs: `GRE 322, admit from Northeastern University, USA, CIBIL 780, age 24, Indian, no collateral.`
2. Agent detects pre-approval intent and extracts all parameters
3. `POST /v1/preapproval` is called against the mock PFL API
4. Pre-approved limit is returned based on GRE score + university tier scorecard
5. LOS and RegIntel JSON payloads are generated for the application journey
6. User uploads PAN card → OCR extracts fields → DQI score is calculated
7. PAN name is cross-validated against University Offer Letter → anomaly flagged if mismatch

---

## Important Disclaimer

This is a **mock demo** built for hackathon purposes. It does not represent an official Poonawalla Fincorp sanction, underwriting rulebook, or production integration. Final approval in a real system would require verified documents, KYC, bureau data, lender policy checks, and live LOS/RegIntel connectivity. The GRE/university scoring grid is intentionally marked as a replaceable demo policy.
