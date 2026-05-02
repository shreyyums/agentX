#!/usr/bin/env python3
"""
OCR & DQI Engine for Indian identity and academic documents.

Examples:
    python ocr_dqi_engine.py samples/pan.jpg --doc-type pan
    python ocr_dqi_engine.py samples/marksheet.pdf --doc-type marksheet --output result.json

The script uses EasyOCR for text extraction, then applies document-specific
parsers and a Data Quality Index (DQI) score over the extracted fields.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PAN_PATTERN = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
DATE_PATTERN = re.compile(r"\b([0-3]?\d[/-][01]?\d[/-](?:19|20)?\d{2})\b")
ROLL_PATTERN = re.compile(r"\b(?:roll|seat|reg(?:istration)?|enrol(?:lment)?)\s*(?:no|number|#)?\s*[:\-]?\s*([A-Z0-9/-]{4,})", re.I)
YEAR_PATTERN = re.compile(r"\b(20\d{2}|19\d{2})\b")
PERCENT_PATTERN = re.compile(r"\b(100(?:\.0+)?|[0-9]{1,2}(?:\.\d+)?)\s*%")
MARKS_PAIR_PATTERN = re.compile(r"\b([0-9]{1,3})\s*/\s*([0-9]{2,3})\b")
OFFER_LETTER_NAME_LABELS = [
    "student name",
    "applicant name",
    "candidate name",
    "name of applicant",
    "name",
]


@dataclass
class OCRLine:
    text: str
    confidence: float
    box: list[list[float]]


def normalize_text(text: str) -> str:
    text = text.replace("|", "I")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def import_easyocr() -> Any:
    try:
        import easyocr  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "EasyOCR is not installed. Install dependencies with:\n"
            "  pip install -r requirements.txt\n\n"
            "For PDF input, also install Poppler and make sure it is on PATH."
        ) from exc
    return easyocr


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("OpenCV is not installed. Run: pip install -r requirements.txt") from exc
    return cv2


def prepare_image(path: Path, temp_dir: Path) -> list[Path]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pdf2image import convert_from_path  # type: ignore
        except ImportError as exc:
            raise SystemExit("PDF support needs pdf2image. Run: pip install -r requirements.txt") from exc

        pages = convert_from_path(str(path), dpi=220)
        image_paths: list[Path] = []
        for index, page in enumerate(pages, start=1):
            page_path = temp_dir / f"page_{index}.png"
            page.save(page_path)
            image_paths.append(page_path)
        return image_paths

    return [path]


def preprocess_image(image_path: Path, temp_dir: Path) -> Path:
    cv2 = import_cv2()
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresholded = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    output = temp_dir / f"preprocessed_{image_path.stem}.png"
    cv2.imwrite(str(output), thresholded)
    return output


def run_ocr(path: Path, languages: list[str], use_gpu: bool, preprocess: bool) -> list[OCRLine]:
    easyocr = import_easyocr()
    reader = easyocr.Reader(languages, gpu=use_gpu)
    lines: list[OCRLine] = []

    with tempfile.TemporaryDirectory(prefix="ocr_dqi_") as tmp:
        temp_dir = Path(tmp)
        image_paths = prepare_image(path, temp_dir)
        for image_path in image_paths:
            ocr_path = preprocess_image(image_path, temp_dir) if preprocess else image_path
            results = reader.readtext(str(ocr_path), detail=1, paragraph=False)
            for box, text, confidence in results:
                cleaned = normalize_text(text)
                if cleaned:
                    lines.append(OCRLine(text=cleaned, confidence=float(confidence), box=box))

    return lines


def joined_text(lines: Iterable[OCRLine]) -> str:
    return "\n".join(line.text for line in lines)


def likely_name(line: str) -> bool:
    if len(line) < 3 or any(char.isdigit() for char in line):
        return False
    blocked = {
        "INCOME TAX DEPARTMENT",
        "GOVT OF INDIA",
        "GOVERNMENT OF INDIA",
        "PERMANENT ACCOUNT NUMBER",
        "FATHER",
        "DATE OF BIRTH",
        "SIGNATURE",
    }
    upper = line.upper()
return upper not in blocked and bool(re.fullmatch(r"[A-Z][A-Z .'-]{2,}", upper))


def extract_pan(lines: list[OCRLine]) -> dict[str, Any]:
    text = joined_text(lines).upper()
    pan = first_match(PAN_PATTERN, text)
    dob = first_match(DATE_PATTERN, text)

    candidates = [line.text.upper() for line in lines if likely_name(line.text)]
    name = candidates[0] if candidates else None
    father_name = candidates[1] if len(candidates) > 1 else None

    return {
        "document_type": "pan",
        "fields": {
            "pan_number": pan,
            "name": name,
            "father_name": father_name,
            "date_of_birth": normalize_date(dob) if dob else None,
        },
    }


def extract_marksheet(lines: list[OCRLine]) -> dict[str, Any]:
    text = joined_text(lines)
    upper_text = text.upper()

    name = extract_labeled_value(text, ["student name", "name of student", "candidate name", "name"])
    roll_number = first_match(ROLL_PATTERN, text)
    year = infer_year(upper_text)
    percentage = infer_percentage(text)
    total_marks = infer_total_marks(text)
    subjects = infer_subject_rows(lines)

    return {
        "document_type": "marksheet",
        "fields": {
            "student_name": name,
            "roll_number": roll_number,
            "year": year,
            "percentage": percentage,
            "total_marks": total_marks,
            "subjects": subjects,
        },
    }


def extract_offer_letter(lines: list[OCRLine]) -> dict[str, Any]:
    text = joined_text(lines)
    name = extract_labeled_value(text, OFFER_LETTER_NAME_LABELS)
    if not name:
        name = infer_offer_letter_salutation_name(text)

    return {
        "document_type": "university_offer_letter",
        "fields": {
            "name": name,
            "program": extract_labeled_value(text, ["program", "course", "degree"]),
            "university": infer_university_name(lines),
        },
    }


def extract_generic(lines: list[OCRLine]) -> dict[str, Any]:
    text = joined_text(lines).upper()
    return {
        "document_type": "generic",
        "fields": {
            "pan_number": first_match(PAN_PATTERN, text),
            "date": normalize_date(first_match(DATE_PATTERN, text) or ""),
            "year": infer_year(text),
        },
    }


def first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def extract_labeled_value(text: str, labels: list[str]) -> str | None:
    for label in labels:
        pattern = re.compile(rf"{re.escape(label)}\s*[:\-]?\s*([A-Z][A-Z .'-]{{2,}})", re.I)
        match = pattern.search(text)
        if match:
            value = re.split(r"\s{2,}|\n|,", match.group(1).strip())[0]
            return clean_person_or_label_value(value)
    return None


def clean_person_or_label_value(value: str) -> str | None:
    value = normalize_text(value).upper()
    value = re.sub(r"\b(?:PROGRAM|COURSE|DEGREE|OFFER|LETTER|DATE)\b.*$", "", value)
    return value.strip(" .'-") or None


def infer_offer_letter_salutation_name(text: str) -> str | None:
    patterns = [
        r"\bDEAR\s+(?:MR\.?|MS\.?|MRS\.?)?\s*([A-Z][A-Z .'-]{2,})",
        r"\bCONGRATULATIONS\s+([A-Z][A-Z .'-]{2,})",
    ]
    upper_text = text.upper()
    for pattern in patterns:
        match = re.search(pattern, upper_text)
        if match:
            value = re.split(r"\s{2,}|\n|,", match.group(1).strip())[0]
            return clean_person_or_label_value(value)
    return None


def infer_university_name(lines: list[OCRLine]) -> str | None:
    for line in lines[:8]:
        upper = line.text.upper()
        if "UNIVERSITY" in upper or "INSTITUTE" in upper or "COLLEGE" in upper:
            return normalize_text(upper)
    return None


def normalize_date(value: str) -> str | None:
    if not value:
        return None
    parts = re.split(r"[/-]", value)
    if len(parts) != 3:
        return value
    day, month, year = parts
    if len(year) == 2:
        year = f"19{year}" if int(year) > 30 else f"20{year}"
    return f"{day.zfill(2)}-{month.zfill(2)}-{year}"


def infer_year(text: str) -> str | None:
    years = [int(match) for match in YEAR_PATTERN.findall(text)]
    valid_years = [year for year in years if 1980 <= year <= 2100]
    return str(max(valid_years)) if valid_years else None


def infer_percentage(text: str) -> float | None:
    percent = first_match(PERCENT_PATTERN, text)
    if percent is not None:
        return round(float(percent), 2)

    marks = infer_total_marks(text)
    if marks and marks.get("obtained") is not None and marks.get("maximum"):
        return round((marks["obtained"] / marks["maximum"]) * 100, 2)
    return None


def infer_total_marks(text: str) -> dict[str, int] | None:
    pairs = [(int(a), int(b)) for a, b in MARKS_PAIR_PATTERN.findall(text)]
    pairs = [(a, b) for a, b in pairs if a <= b and b >= 50]
    if not pairs:
        return None
    obtained = sum(pair[0] for pair in pairs)
    maximum = sum(pair[1] for pair in pairs)
    return {"obtained": obtained, "maximum": maximum}


def infer_subject_rows(lines: list[OCRLine]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subject_pattern = re.compile(r"^([A-Z][A-Z &.-]{2,})\s+([0-9]{1,3})(?:\s*/\s*|\s+)([0-9]{2,3})\b", re.I)
    for line in lines:
        match = subject_pattern.search(line.text)
        if not match:
            continue
        obtained = int(match.group(2))
        maximum = int(match.group(3))
        if obtained <= maximum:
            rows.append(
                {
                    "subject": normalize_text(match.group(1)).upper(),
                    "obtained": obtained,
                    "maximum": maximum,
                    "confidence": round(line.confidence, 3),
                }
            )
    return rows


def compute_dqi(document_type: str, fields: dict[str, Any], lines: list[OCRLine]) -> dict[str, Any]:
    required = {
        "pan": ["pan_number", "name", "father_name", "date_of_birth"],
        "marksheet": ["student_name", "roll_number", "year", "percentage"],
        "generic": [],
    }.get(document_type, [])

    missing = [field for field in required if not fields.get(field)]
    validations = validate_fields(document_type, fields)
    invalid = [name for name, passed in validations.items() if not passed]
    mean_confidence = statistics.mean([line.confidence for line in lines]) if lines else 0.0
    completeness = (len(required) - len(missing)) / len(required) if required else 1.0
    validity = (len(validations) - len(invalid)) / len(validations) if validations else 1.0

    score = round((0.45 * completeness + 0.35 * validity + 0.20 * mean_confidence) * 100, 2)
    return {
        "score": score,
        "grade": grade_dqi(score),
        "ocr_confidence": round(mean_confidence, 3),
        "completeness": round(completeness, 3),
        "validity": round(validity, 3),
        "missing_fields": missing,
        "failed_validations": invalid,
        "validation_results": validations,
    }


def validate_fields(document_type: str, fields: dict[str, Any]) -> dict[str, bool]:
    validations: dict[str, bool] = {}
    if document_type == "pan":
        pan_number = fields.get("pan_number") or ""
        validations["pan_format"] = bool(PAN_PATTERN.fullmatch(pan_number))
        validations["date_of_birth_format"] = bool(re.fullmatch(r"[0-3]\d-[01]\d-(?:19|20)\d{2}", fields.get("date_of_birth") or ""))
    elif document_type == "marksheet":
        percentage = fields.get("percentage")
        validations["percentage_range"] = percentage is None or 0 <= float(percentage) <= 100
        total_marks = fields.get("total_marks")
        validations["marks_range"] = not total_marks or total_marks["obtained"] <= total_marks["maximum"]
    return validations


def grade_dqi(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    return "D"


def normalize_name_for_match(name: str | None) -> str:
    if not name:
        return ""
    name = name.upper()
    name = re.sub(r"\b(?:MR|MS|MRS|DR|SHRI|SMT|KUMARI)\b\.?", " ", name)
    name = re.sub(r"[^A-Z ]", " ", name)
    return normalize_text(name)


def name_similarity(left: str | None, right: str | None) -> float:
    left_normalized = normalize_name_for_match(left)
    right_normalized = normalize_name_for_match(right)
    if not left_normalized or not right_normalized:
        return 0.0
    return round(difflib.SequenceMatcher(None, left_normalized, right_normalized).ratio(), 3)


def build_agentic_dqi(
    pan_result: dict[str, Any],
    offer_letter_path: Path,
    offer_letter_lines: list[OCRLine],
    name_threshold: float,
) -> dict[str, Any]:
    offer_letter = extract_offer_letter(offer_letter_lines)
    pan_name = pan_result.get("fields", {}).get("name")
    offer_name = offer_letter["fields"].get("name")
    similarity = name_similarity(pan_name, offer_name)
    anomalies: list[dict[str, Any]] = []

    if not pan_name or not offer_name:
        anomalies.append(
            {
                "type": "Data Quality Anomaly",
                "rule": "PAN name must be comparable with University Offer Letter name",
                "severity": "medium",
                "message": "Could not compare names because one document is missing an extracted name.",
                "pan_name": pan_name,
                "offer_letter_name": offer_name,
            }
        )
    elif similarity < name_threshold:
        anomalies.append(
            {
                "type": "Data Quality Anomaly",
                "rule": "PAN name must match University Offer Letter name",
                "severity": "high",
                "message": "Extracted PAN name does not match the University Offer Letter name.",
                "pan_name": pan_name,
                "offer_letter_name": offer_name,
                "similarity": similarity,
                "threshold": name_threshold,
            }
        )

    return {
        "status": "anomaly_detected" if anomalies else "passed",
        "offer_letter_source_file": str(offer_letter_path),
        "offer_letter_fields": offer_letter["fields"],
        "checks": {
            "pan_offer_letter_name_match": {
                "passed": not anomalies,
                "pan_name": pan_name,
                "offer_letter_name": offer_name,
                "similarity": similarity,
                "threshold": name_threshold,
            }
        },
        "anomalies": anomalies,
    }


def detect_doc_type(lines: list[OCRLine]) -> str:
    text = joined_text(lines).upper()
    if PAN_PATTERN.search(text) or "INCOME TAX" in text or "PERMANENT ACCOUNT" in text:
        return "pan"
    if any(word in text for word in ["MARKSHEET", "MARK SHEET", "STATEMENT OF MARKS", "ROLL NO", "SUBJECT"]):
        return "marksheet"
    return "generic"


def build_result(path: Path, doc_type: str, lines: list[OCRLine]) -> dict[str, Any]:
    actual_type = detect_doc_type(lines) if doc_type == "auto" else doc_type
    if actual_type == "pan":
        extracted = extract_pan(lines)
    elif actual_type == "marksheet":
        extracted = extract_marksheet(lines)
    else:
        extracted = extract_generic(lines)

    fields = extracted["fields"]
    return {
        "source_file": str(path),
        "document_type": extracted["document_type"],
        "fields": fields,
        "dqi": compute_dqi(extracted["document_type"], fields, lines),
        "ocr": [
            {"text": line.text, "confidence": round(line.confidence, 3), "box": line.box}
            for line in lines
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract key data and DQI from PAN cards and marksheets.")
    parser.add_argument("document", type=Path, help="Path to an image or PDF document.")
    parser.add_argument("--doc-type", choices=["auto", "pan", "marksheet", "generic"], default="auto")
    parser.add_argument("--languages", nargs="+", default=["en"], help="EasyOCR languages, e.g. en hi.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for EasyOCR if available.")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable OpenCV preprocessing.")
    parser.add_argument("--offer-letter", type=Path, help="Optional University Offer Letter to compare against a PAN card.")
    parser.add_argument("--name-match-threshold", type=float, default=0.82, help="Similarity threshold for PAN vs offer-letter name match.")
    parser.add_argument("--output", type=Path, help="Optional JSON output file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.document.exists():
        raise SystemExit(f"File not found: {args.document}")

    lines = run_ocr(
        path=args.document,
        languages=args.languages,
        use_gpu=args.gpu,
        preprocess=not args.no_preprocess,
    )
    result = build_result(args.document, args.doc_type, lines)
    if args.offer_letter:
        if not args.offer_letter.exists():
            raise SystemExit(f"Offer letter not found: {args.offer_letter}")
        if result["document_type"] != "pan":
            raise SystemExit("--offer-letter comparison is only supported when the main document is a PAN card.")
        offer_letter_lines = run_ocr(
            path=args.offer_letter,
            languages=args.languages,
            use_gpu=args.gpu,
            preprocess=not args.no_preprocess,
        )
        result["agentic_dqi"] = build_agentic_dqi(
            pan_result=result,
            offer_letter_path=args.offer_letter,
            offer_letter_lines=offer_letter_lines,
            name_threshold=args.name_match_threshold,
        )
    payload = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
