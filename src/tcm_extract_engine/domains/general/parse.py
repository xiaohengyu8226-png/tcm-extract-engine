from typing import Any

from tcm_extract_engine.core.records import SourceRecord, SourceRef


def parse_record(
    raw: dict[str, Any],
    *,
    book_id: str,
    seq: int,
    extractor: str,
    page_fallback: int | None,
    heading_path: list[str],
    title: str,
) -> SourceRecord:
    record_type = str(
        raw.get("record_type") or ("FIGURE" if extractor == "FigureExtractor" else "KNOWLEDGE")
    ).upper()
    payload = raw.get("payload")
    if payload is None:
        if record_type == "FIGURE":
            payload = {
                "image_ref": raw.get("image_ref", {}),
                "attributes": raw.get("attributes", {}),
            }
        else:
            payload = {
                k: raw[k]
                for k in (
                    "kind",
                    "subject",
                    "predicate",
                    "object",
                    "conditions",
                    "scope",
                    "negated",
                    "modality",
                    "exceptions",
                    "quantities",
                    "time",
                    "steps",
                    "attributes",
                )
                if k in raw
            }
            payload.setdefault("subject", raw.get("subject_text", ""))
    page = raw.get("page_hint", page_fallback)
    try:
        page = int(page) if page is not None else None
    except (ValueError, TypeError):
        page = page_fallback
    return SourceRecord(
        record_id=f"SR-{book_id[:8]}-{seq:06d}",
        book_id=book_id,
        domain_id="general",
        record_type=record_type,
        source=SourceRef(page_start=page, page_end=page, section_path=heading_path or [title]),
        raw_text=str(raw.get("raw_text") or ""),
        subject_text=raw.get("subject_text") or payload.get("subject"),
        payload=payload,
        ambiguities=[str(a) for a in raw.get("ambiguities", [])],
    )
