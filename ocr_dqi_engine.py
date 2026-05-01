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
