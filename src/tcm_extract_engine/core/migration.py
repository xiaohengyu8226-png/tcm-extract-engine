"""Explicit, lossless legacy-to-envelope migration; no automatic read coercion."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from tcm_extract_engine.core.artifacts import _atomic_write_json, artifact_hash
from tcm_extract_engine.core.records import SourceRecord

LEGACY_FIELDS = frozenset(
    {
        "herb_name",
        "toxicity_level",
        "dose",
        "interaction",
        "pregnancy",
        "population",
        "herb_facts",
        "image_ref",
        "action",
        "attributes",
    }
)


def migrate_record(raw: dict[str, Any], domain_id: str = "tcm") -> dict[str, Any]:
    if raw.get("schema_version") == "2":
        return SourceRecord.model_validate(raw).model_dump(mode="json")
    if domain_id != "tcm":
        raise ValueError("旧 typed 工件只能按 tcm 无损迁移；切换 general 应从 profile 重抽")
    envelope = {k: v for k, v in raw.items() if k not in LEGACY_FIELDS}
    payload = {k: v for k, v in raw.items() if k in LEGACY_FIELDS}
    if raw.get("record_type") == "TABLE_ROW":
        nonempty = {
            k: v for k, v in payload.items() if v not in (None, {}, [], "") and k != "attributes"
        }
        if nonempty:
            raise ValueError(f"TABLE_ROW 有非预期领域字段，拒绝丢弃: {list(nonempty)}")
        payload = {"attributes": payload.get("attributes", {})}
    envelope.update(schema_version="2", domain_id=domain_id, payload=payload)
    return SourceRecord.model_validate(envelope).model_dump(mode="json")


def migrate_book(book_dir: Path, *, dry_run: bool = True) -> dict[str, Any]:
    manifest_path = book_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    domain_id = manifest.get("domain_id", "tcm")
    changes: dict[Path, str] = {}
    counts = {}
    for relative in [
        "04_extract/records.jsonl",
        "06_verify/verified_records.jsonl",
        "05_validate/suspicious_records.jsonl",
    ]:
        path = book_dir / relative
        if not path.is_file():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        before = [r.get("record", r) for r in rows if "header" not in r]
        after = [migrate_record(r, domain_id) for r in before]
        counts[relative] = {
            "before": len(before),
            "after": len(after),
            "type_counts_equal": sorted(r["record_type"] for r in before)
            == sorted(r["record_type"] for r in after),
        }
        if any(r.get("schema_version") != "2" for r in before):
            head = (
                rows[0]
                if rows and "header" in rows[0]
                else {
                    "header": {
                        "schema_version": "1",
                        "book_id": manifest["book_id"],
                        "task_id": "migration",
                    }
                }
            )
            head["header"].setdefault("extra", {}).update(
                record_schema_version="2", domain_id=domain_id, pack_version="2"
            )
            changes[path] = "".join(
                json.dumps(r, ensure_ascii=False) + "\n" for r in [head, *after]
            )
    correction_path = book_dir / "06_verify/correction_log.jsonl"
    if correction_path.is_file():
        rows = [
            json.loads(line) for line in correction_path.read_text().splitlines() if line.strip()
        ]
        changed = False
        for wrapper in rows:
            row = wrapper.get("record", wrapper)
            field = row.get("field")
            if field in LEGACY_FIELDS:
                row["field"] = "/payload/" + field
                changed = True
            elif field in {"raw_text", "subject_text"}:
                row["field"] = "/" + field
                changed = True
        if changed:
            changes[correction_path] = "".join(
                json.dumps(r, ensure_ascii=False) + "\n" for r in rows
            )
    report = {
        "book_id": manifest["book_id"],
        "dry_run": dry_run,
        "counts": counts,
        "changed_files": [str(p.relative_to(book_dir)) for p in changes],
        "derived_artifacts_stale": bool(changes),
        "old_caches": "ignored_by_version",
    }
    if not changes or dry_run:
        return report
    # Validate the entire migration before any writes; backups are never overwritten.
    for path in [*changes, manifest_path]:
        backup = path.with_name(path.name + ".envelope-v1.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
    _atomic_write_json(book_dir / "envelope_migration.json", {**report, "status": "WRITING"})
    for path, content in changes.items():
        temp = path.with_suffix(path.suffix + ".migrating")
        temp.write_text(content, encoding="utf-8")
        temp.replace(path)
    manifest["domain_id"] = domain_id
    manifest["record_schema_version"] = "2"
    # Recompute fingerprints of changed stage artifacts; derived stages need regeneration.
    for stage, relative in [
        ("extract", "04_extract/records.jsonl"),
        ("verify", "06_verify/verified_records.jsonl"),
    ]:
        path = book_dir / relative
        if path in changes and stage in manifest.get("stages", {}):
            data = [json.loads(s) for s in path.read_text().splitlines() if s.strip()][1:]
            manifest["stages"][stage]["artifact_hash"] = artifact_hash(data)
    for stage in ["validate", "verify", "review", "publish"]:
        manifest.get("stages", {}).pop(stage, None)
    manifest["derived_artifacts_stale"] = True
    if manifest.get("status") in {"VALIDATING", "VALIDATED", "VERIFYING", "VERIFIED", "WAITING_REVIEW", "REVIEWED", "PUBLISHED"}:
        manifest["status"] = "EXTRACTED"
        manifest["error"] = None
    _atomic_write_json(manifest_path, manifest)
    _atomic_write_json(book_dir / "envelope_migration.json", {**report, "status": "DONE"})
    return report
