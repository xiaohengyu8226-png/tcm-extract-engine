"""Deterministic evidence-rich handoff. No graph fusion or implicit publication."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from tcm_extract_engine.cab.profiler import load_blocks
from tcm_extract_engine.core.artifacts import (
    _atomic_write_json,
    artifact_hash,
    load_artifact_records,
)
from tcm_extract_engine.core.records import SourceRecord
from tcm_extract_engine.domains import get_domain_pack


class ExportedRecord(SourceRecord):
    verification: str = "UNVERIFIED"
    usable: bool = False


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    temp.replace(path)


def export_knowledge(book_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    target = output_dir or book_dir / "08_publish" / "knowledge"
    target.mkdir(parents=True, exist_ok=True)
    blocks = load_blocks(book_dir / "01_parse/blocks.jsonl")
    by_id = {str(b["blockid"]): b for b in blocks}
    manifest = json.loads((book_dir / "manifest.json").read_text())
    verified = book_dir / "06_verify/verified_records.jsonl"
    verified_current = manifest.get("stages", {}).get("verify", {}).get("status") == "DONE" and not manifest.get("derived_artifacts_stale", False)
    if verified_current:
        if not verified.is_file():
            raise ValueError("manifest 声明核验完成但核验工件缺失")
        from tcm_extract_engine.runner import _artifact_digest
        validate_path = book_dir / "05_validate/validation_report.json"
        extract_path = book_dir / "04_extract/records.jsonl"
        header = json.loads(verified.read_text().splitlines()[0]).get("header", {})
        validation = json.loads(validate_path.read_text())
        if header.get("upstream_hash") != _artifact_digest(validate_path) or validation.get("header", {}).get("upstream_hash") != _artifact_digest(extract_path):
            raise ValueError("核验工件上游已变化，请重跑 validate/verify")
    source_path = verified if verified_current else book_dir / "04_extract/records.jsonl"
    records = load_artifact_records(source_path, SourceRecord)
    verdict_path = book_dir / "06_verify/verification_results.jsonl"
    verdicts = {}
    if verified_current and verdict_path.is_file():
        for line in verdict_path.read_text().splitlines():
            row = json.loads(line)
            if "record_id" in row:
                verdicts[row["record_id"]] = row
    pack = get_domain_pack(manifest.get("domain_id", "tcm"))
    evidence: dict[str, dict[str, Any]] = {}
    assets: dict[str, dict[str, Any]] = {}
    rows = []
    quarantine = []
    validation_path = book_dir / "05_validate/validation_report.json"
    validation_data = json.loads(validation_path.read_text()) if validation_path.is_file() else {}
    invalid_ids = {item["record_id"] for item in validation_data.get("issues", []) if item.get("severity") == "ERROR"}
    visual_items = []
    missing = []
    for record in records:
        if record.domain_id != pack.id:
            raise ValueError("工件领域与 manifest 不一致，禁止导出")
        missing_for_record = False
        record.source.evidence_ids = []
        for block_id in record.source.block_ids:
            block = by_id.get(block_id)
            if block is None:
                missing_for_record = True
                missing.append({"record_id": record.record_id, "block_id": block_id})
                continue
            content = str(block.get("content") or "")
            positions = block.get("positions") or []
            page = record.source.page_start
            if positions:
                try:
                    page = int(positions[0].get("anchor"))
                except (ValueError, TypeError):
                    pass
            basis = {
                "book": record.book_id,
                "block": block_id,
                "quote": record.raw_text,
                "row": record.payload.get("row_index") if record.passthrough else None,
            }
            evidence_id = "EV-" + artifact_hash(basis).split(":")[1][:24]
            span = next((s for s in getattr(record.source, "evidence_spans", []) if s["block_id"] == block_id), None)
            quote = span["quote"] if span else record.raw_text
            exact = bool(quote) and quote in content
            entry = {
                "evidence_id": evidence_id,
                "book_id": record.book_id,
                "block_id": block_id,
                "page": page,
                "section_path": record.source.section_path,
                "quote": quote if exact else None,
                "block_content": content,
                "location_granularity": "character" if exact else "block",
                "quote_match": exact,
                "positions": positions,
                "coordinate_system": "parser_native",
            }
            if exact:
                entry["char_start"] = content.index(quote)
                entry["char_end"] = entry["char_start"] + len(quote)
                entry["text_version"] = hashlib.sha256(content.encode()).hexdigest()
            if record.passthrough:
                entry["table"] = {
                    "table_id": block_id,
                    "row_index": record.payload.get("row_index"),
                    "headers": record.payload.get("headers", []),
                    "cells": record.payload.get("cells", []),
                }
            asset_path = Path(str(block.get("asset_path") or ""))
            if block.get("asset_path") and asset_path.is_file():
                digest = hashlib.sha256(asset_path.read_bytes()).hexdigest()
                asset_id = "AS-" + digest
                relative = Path("assets") / (digest + asset_path.suffix.lower())
                dest = target / relative
                dest.parent.mkdir(exist_ok=True)
                if not dest.is_file():
                    shutil.copy2(asset_path, dest)
                assets[asset_id] = {
                    "asset_id": asset_id,
                    "sha256": digest,
                    "path": str(relative),
                    "media_type": "image",
                    "block_ids": sorted(
                        set(assets.get(asset_id, {}).get("block_ids", []) + [block_id])
                    ),
                }
                entry["asset_id"] = asset_id
            evidence[evidence_id] = entry
            record.source.evidence_ids.append(evidence_id)
        verdict = verdicts.get(record.record_id, {}).get("verdict", "UNVERIFIED")
        row = record.model_dump(mode="json")
        # Asset paths on handoff refer to packaged content, never machine-specific paths.
        image = row["payload"].get("image_ref")
        if image:
            refs: list[str] = [
                str(evidence[e]["asset_id"])
                for e in record.source.evidence_ids
                if evidence[e].get("asset_id")
            ]
            image["asset_path"] = assets[refs[0]]["path"] if refs else ""
            image["asset_id"] = refs[0] if refs else None
        row["verification"] = verdict
        row["usable"] = (
            bool(record.source.evidence_ids)
            and not missing_for_record
            and (not image or bool(refs))
            and not record.needs_review
            and verdict == "SUPPORTED"
            and not record.passthrough
        )
        quarantined = not record.source.evidence_ids or missing_for_record or record.record_id in invalid_ids or bool(image and not refs)
        if quarantined:
            row["usable"] = False
        if image and not quarantined:
            for kind, key in [("observation", "facts"), ("process_step", "steps"), ("diagram_relation", "relations"), ("decision", "decision_points"), ("measurement", "data_points")]:
                for item in image.get("extracted_data", {}).get(key, []):
                    item_id = "VF-" + artifact_hash({"parent": record.record_id, "kind": kind, "item": item}).split(":")[1][:24]
                    visual_items.append({"fact_id": item_id, "parent_figure_id": record.record_id,
                        "book_id": record.book_id, "kind": kind, "content": item,
                        "evidence_ids": record.source.evidence_ids, "asset_id": image.get("asset_id"),
                        "verification": verdict, "verification_scope": "parent_figure_all_fields",
                        "usable": row["usable"]})
        normalized = ExportedRecord.model_validate(row).model_dump(mode="json")
        (quarantine if quarantined else rows).append(normalized)
    plan_path = book_dir / "03_plan/extraction_plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else {}
    planned = [s for s in plan.get("sections", []) if not s.get("skip_reason")]
    planned_ids = {str(b) for s in planned for b in s.get("block_ids", [])}
    covered = {b for rec in records for b in (rec.source.context_block_ids or rec.source.block_ids)}
    # No-record batches may be successfully processed. Prefer extraction batch ledger.
    ledger_path = book_dir / "04_extract/batch_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.is_file() else []
    if ledger:
        covered = {
            b for item in ledger if item.get("status") == "DONE" for b in item.get("block_ids", [])
        }
    report = {
        "book_id": manifest["book_id"],
        "extracted_records": len(records),
        "records": len(rows),
        "quarantined_records": len(quarantine),
        "usable_records": sum(r["usable"] for r in rows),
        "verdict_counts": dict(Counter(r["verification"] for r in rows)),
        "evidence_count": len(evidence),
        "missing_references": missing,
        "unlocated_records": [r["record_id"] for r in quarantine if not r["source"]["evidence_ids"]],
        "planned_blocks": len(planned_ids),
        "processed_blocks": len(planned_ids & covered),
        "unprocessed_block_ids": sorted(planned_ids - covered),
        "processing_coverage_source": "batch_ledger" if ledger else "record_context_only",
        "precision": None,
        "reference_recall": None,
        "gold_status": "pending_human_confirmation",
        "human_review_status": "not_approved",
    }
    _write_jsonl(target / "records.jsonl", rows)
    _write_jsonl(target / "quarantine.jsonl", quarantine)
    _write_jsonl(target / "usable_records.jsonl", [r for r in rows if r["usable"]])
    _write_jsonl(target / "evidence.jsonl", list(evidence.values()))
    _write_jsonl(target / "assets.jsonl", list(assets.values()))
    _write_jsonl(target / "visual_facts.jsonl", visual_items)
    _atomic_write_json(target / "quality_report.json", report)
    _atomic_write_json(
        target / "run_manifest.json",
        {
            "schema_version": "2",
            "book": manifest,
            "domain_id": pack.id,
            "pack_version": pack.version,
            "records_hash": artifact_hash(rows),
            "evidence_hash": artifact_hash(list(evidence.values())),
            "source_artifact": str(source_path.relative_to(book_dir)),
            "scope": "book-local extraction; no entity alignment or graph fusion",
        },
    )
    return report
