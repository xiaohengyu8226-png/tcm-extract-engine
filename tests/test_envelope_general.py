"""Behavior contracts for generic extraction, migration, evidence and visual checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tcm_extract_engine.cab.extractors import extract_records, split_section_batches
from tcm_extract_engine.cab.planner import SectionPlan
from tcm_extract_engine.cab.verifier import CorrectionEntry, apply_correction, verify_records
from tcm_extract_engine.core.artifacts import ArtifactHeader, save_artifact
from tcm_extract_engine.core.export import export_knowledge
from tcm_extract_engine.core.llm import LLMUsage
from tcm_extract_engine.core.migration import migrate_book, migrate_record
from tcm_extract_engine.core.records import SourceRecord, stable_record_id
from tcm_extract_engine.domains import get_domain_pack, registered_domains
from tcm_extract_engine.gepa.bridge import evaluate_candidate


def fact(**overrides: Any) -> SourceRecord:
    return SourceRecord(
        **{
            "record_id": "r1",
            "book_id": "b",
            "domain_id": "general",
            "record_type": "KNOWLEDGE",
            "payload": {
                "subject": "材料A",
                "predicate": "软化温度为",
                "object": "80℃",
                "conditions": ["常压下"],
            },
            "raw_text": "常压下材料A的软化温度为80℃。",
            "source": {"block_ids": ["b1"], "page_start": 1},
            **overrides,
        }
    )


def test_core_import_does_not_load_tcm_schema() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys; from tcm_extract_engine.core.records import SourceRecord; assert "tcm_extract_engine.domains.tcm.schema" not in sys.modules',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_legacy_disk_record_requires_explicit_migration() -> None:
    with pytest.raises(ValidationError):
        SourceRecord.model_validate(
            {"record_id": "r", "book_id": "b", "record_type": "DOSAGE", "dose": {"min_dose": 1}}
        )
    raw = {
        "record_id": "r",
        "book_id": "b",
        "record_type": "DOSAGE",
        "raw_text": "剂量1~3g",
        "dose": {"min_dose": 1, "max_dose": 3},
        "herb_name": "某药",
        "attributes": {"note": "先煎"},
    }
    migrated = migrate_record(raw)
    assert migrated["record_id"] == "r"
    assert migrated["payload"]["dose"]["min_dose"] == 1
    assert migrated["payload"]["attributes"]["note"] == "先煎"
    assert migrate_record(migrated) == migrated


def test_general_schema_rejects_unknown_type_and_missing_fact() -> None:
    with pytest.raises(ValidationError):
        fact(record_type="DOSAGE")
    with pytest.raises(ValidationError):
        fact(payload={"subject": "材料", "predicate": "性质"})
    original = fact()
    loaded = SourceRecord.model_validate_json(original.model_dump_json())
    assert loaded.payload["conditions"] == ["常压下"]
    assert set(registered_domains()) >= {"general", "tcm"}


def test_record_identity_stable_and_changes_with_fact() -> None:
    record = fact()
    original = stable_record_id(record)
    record.record_id = "unrelated-sequence"
    record.needs_review = True
    assert stable_record_id(record) == original
    record.payload["conditions"] = ["高压下"]
    assert stable_record_id(record) != original


def test_general_mixed_section_preserves_text_route(tmp_path: Path) -> None:
    pack = get_domain_pack("general")
    assert pack.route_section("实验结果", table_count=1) == "KNOWLEDGE"
    section = SectionPlan(
        section_id="s",
        title="实验结果",
        extractor="KnowledgeExtractor",
        template="knowledge.md",
        prompt_version="x",
        block_ids=["t", "p", "i"],
    )
    batches = split_section_batches(
        section,
        {
            "t": {
                "content": "<table><tr><td>A</td></tr><tr><td>B</td></tr></table>",
                "format": "drawing",
            },
            "p": {"content": "材料性质"},
            "i": {"content": "示意图", "format": "drawing", "asset_path": "image.png"},
        },
        pack=pack,
    )
    assert any(b.passthrough for b in batches)
    assert any(b.multimodal for b in batches)
    assert any(
        not b.passthrough and not b.multimodal and b.section.template == "knowledge.md"
        for b in batches
    )


def test_correction_checks_path_original_and_payload() -> None:
    record = fact()
    correction = CorrectionEntry(
        record_id="r1", field="/payload/object", original="80℃", suggested="90℃"
    )
    corrected = apply_correction(record, correction)
    assert corrected.payload["object"] == "90℃"
    assert record.payload["object"] == "80℃"
    with pytest.raises(ValueError, match="原值"):
        apply_correction(corrected, correction)
    invalid = correction.model_copy(
        update={"field": "/payload/subject", "original": "材料A", "suggested": ""}
    )
    with pytest.raises(ValidationError):
        apply_correction(record, invalid)


def test_migration_dry_run_and_backup(tmp_path: Path) -> None:
    (tmp_path / "04_extract").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"book_id": "b", "stages": {}}))
    target = tmp_path / "04_extract/records.jsonl"
    text = (
        json.dumps({"header": {"schema_version": "1", "book_id": "b", "task_id": "old"}})
        + "\n"
        + json.dumps(
            {
                "record_id": "r",
                "book_id": "b",
                "record_type": "GENERAL_RULE",
                "attributes": {"x": "y"},
            }
        )
        + "\n"
    )
    target.write_text(text)
    report = migrate_book(tmp_path)
    assert report["changed_files"]
    assert target.read_text() == text
    migrate_book(tmp_path, dry_run=False)
    backup = target.with_name(target.name + ".envelope-v1.bak")
    assert backup.read_text() == text
    assert not migrate_book(tmp_path, dry_run=False)["changed_files"]
    assert backup.read_text() == text


def write_case(tmp_path: Path, *, images: bool = False) -> tuple[Path, Path]:
    blocks = tmp_path / "blocks.jsonl"
    rows = [
        {"type": "meta"},
        {
            "type": "content",
            "blockid": "b1",
            "content": "常压下材料A的软化温度为80℃。",
            "heading": "实验",
            "positions": [{"anchor": "1"}],
        },
    ]
    if images:
        from PIL import Image

        path = tmp_path / "figure.png"
        Image.new("RGB", (80, 80), "white").save(path)
        rows.append(
            {
                "type": "content",
                "blockid": "i1",
                "content": "流程图",
                "format": "drawing",
                "asset_path": str(path),
                "positions": [{"anchor": "2"}],
            }
        )
    blocks.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    plan = tmp_path / "plan.json"
    section = SectionPlan(
        section_id="s",
        title="实验",
        heading_path=["实验"],
        extractor="KnowledgeExtractor",
        template="knowledge.md",
        prompt_version="old-plan-hash",
        block_ids=["b1"] + (["i1"] if images else []),
    )
    save_artifact(
        plan,
        header=ArtifactHeader(schema_version="1", book_id="b", task_id="t"),
        data={"domain_id": "general", "sections": [section.model_dump(mode="json")]},
    )
    return blocks, plan


class Client:
    model = "fixture"

    def __init__(self, *, count: int = 1) -> None:
        self.usage = LLMUsage()
        self.count = count
        self.visual_calls = 0

    async def complete_json(self, *, system_prompt: str, user_prompt: str, **kwargs: Any) -> dict:
        self.usage.calls += 1
        if "核验器" in system_prompt:
            import re

            ids = list(dict.fromkeys(re.findall(r'"record_id":\s*"([^"]+)"', user_prompt)))
            return {"records": [{"record_id": rid, "verdict": "SUPPORTED"} for rid in ids]}
        return {
            "records": [
                {**fact().model_dump(mode="json"), "record_id": f"r{i}", "subject_text": f"材料{i}"}
                for i in range(self.count)
            ]
        }

    async def complete_multimodal_json(self, *, user_contents: list, **kwargs: Any) -> dict:
        self.visual_calls += 1
        prompt = user_contents[0].text
        if "原图证据核验" in prompt:
            return {
                "records": [{"record_id": "fig", "verdict": "PARTIAL", "needs_human_review": True}]
            }
        return {
            "records": [
                {
                    "record_type": "FIGURE",
                    "raw_text": "流程图",
                    "image_ref": {
                        "asset_path": "",
                        "figure_type": "流程图",
                        "description": "A到B",
                        "extracted_data": {"steps": ["A", "B"]},
                    },
                }
            ]
        }


@pytest.mark.asyncio
async def test_general_real_pipeline_and_stable_cache(tmp_path: Path) -> None:
    blocks, plan = write_case(tmp_path)
    client = Client()
    args = dict(
        book_id="b",
        blocks_path=blocks,
        plan_path=plan,
        cache_dir=tmp_path / "cache",
        client=client,
        pack=get_domain_pack("general"),
    )
    first = await extract_records(**args)
    second = await extract_records(**args)
    assert client.usage.calls == 1
    assert first[0].record_id == second[0].record_id
    assert first[0].source.block_ids == ["b1"]
    assert first[0].generation.template == "knowledge.md"
    assert first[0].generation.prompt_version != "old-plan-hash"


@pytest.mark.asyncio
async def test_gepa_only_assessed_records_in_denominator(tmp_path: Path) -> None:
    blocks, plan = write_case(tmp_path)
    outcome = await evaluate_candidate(
        book_id="b",
        blocks_path=blocks,
        plan_path=plan,
        target_template="knowledge.md",
        constraint_block="",
        extract_client=Client(count=4),
        verify_client=Client(),
        work_dir=tmp_path / "cand",
        verify_sample=1,
    )
    assert outcome.records_in_scope == 4
    assert outcome.evaluated_records == 1
    assert outcome.unassessed_records == 3
    assert outcome.fitness.issue_stats.total_candidates == 1


@pytest.mark.asyncio
async def test_visual_verifier_uses_original_image(tmp_path: Path) -> None:
    blocks, _ = write_case(tmp_path, images=True)
    record = SourceRecord(
        record_id="fig",
        book_id="b",
        domain_id="general",
        record_type="FIGURE",
        raw_text="流程图",
        source={"block_ids": ["i1"]},
        payload={"image_ref": {"asset_path": "untrusted-output.png", "figure_type": "流程图"}},
    )
    vision = Client()
    report = await verify_records(
        records=[record], blocks_path=blocks, client=Client(), vision_client=vision
    )
    assert vision.visual_calls == 1
    assert report.results[0].verdict == "PARTIAL"
    assert record.needs_review


def test_export_retains_all_evidence_and_review_state(tmp_path: Path) -> None:
    blocks, _ = write_case(tmp_path)
    (tmp_path / "01_parse").mkdir()
    (tmp_path / "01_parse/blocks.jsonl").write_text(
        blocks.read_text()
        + json.dumps(
            {"type": "content", "blockid": "b2", "content": "图注", "positions": [{"anchor": "2"}]}
        )
        + "\n"
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"book_id": "b", "domain_id": "general"}))
    (tmp_path / "04_extract").mkdir()
    record = fact(source={"block_ids": ["b1", "b2"]})
    save_artifact(
        tmp_path / "04_extract/records.jsonl",
        header=ArtifactHeader(schema_version="1", book_id="b", task_id="t"),
        data={},
        records=[record],
    )
    report = export_knowledge(tmp_path)
    assert report["evidence_count"] == 2
    assert report["usable_records"] == 0  # Extraction alone never implies verification.
    assert not report["missing_references"]


def test_cross_page_quote_retains_both_evidence_spans() -> None:
    from tcm_extract_engine.cab.extractors import SectionChunk, _attribute_batch
    section = SectionPlan(section_id='s', title='实验', extractor='KnowledgeExtractor', template='knowledge.md', prompt_version='v')
    batch = SectionChunk(section=section, texts=['常压下材料A的', '软化温度为80℃。'], pages=[1, 2], block_ids=['b1', 'b2'])
    record = fact()
    _attribute_batch([record], batch)
    assert record.source.block_ids == ['b1', 'b2']
    assert record.source.page_start == 1 and record.source.page_end == 2
    assert [s['quote'] for s in record.source.evidence_spans] == batch.texts
    assert not record.needs_review


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', [{}, {'records': {}}, {'records': [None]}])
async def test_malformed_response_is_failed_batch(tmp_path: Path, bad: dict) -> None:
    from tcm_extract_engine.cab.extractors import ExtractionError
    blocks, plan = write_case(tmp_path)
    class Broken(Client):
        async def complete_json(self, **kwargs: Any) -> dict:
            return bad
    with pytest.raises(ExtractionError):
        await extract_records(book_id='b', blocks_path=blocks, plan_path=plan,
            cache_dir=tmp_path/'cache', client=Broken())
    ledger = json.loads((tmp_path/'batch_ledger.json').read_text())
    assert ledger[0]['status'] == 'FAILED'


def test_b_cache_includes_full_model_image_and_source_identity() -> None:
    from tcm_extract_engine.cab.verifier import _batch_cache_key
    record = fact()
    baseline = _batch_cache_key([(record, 'source')], '123456789abc:model-a:image-a')
    assert baseline != _batch_cache_key([(record, 'source')], '123456789abc:model-b:image-a')
    assert baseline != _batch_cache_key([(record, 'source')], '123456789abc:model-a:image-b')
    assert baseline != _batch_cache_key([(record, 'different')], '123456789abc:model-a:image-a')


def test_stale_verified_file_never_overrides_current_extract(tmp_path: Path) -> None:
    blocks, _ = write_case(tmp_path)
    (tmp_path/'01_parse').mkdir()
    (tmp_path/'01_parse/blocks.jsonl').write_text(blocks.read_text())
    (tmp_path/'04_extract').mkdir()
    (tmp_path/'06_verify').mkdir()
    (tmp_path/'manifest.json').write_text(json.dumps({'book_id':'b','domain_id':'general','stages':{'extract':{'status':'DONE'}}}))
    header=ArtifactHeader(schema_version='1',book_id='b',task_id='test')
    save_artifact(tmp_path/'04_extract/records.jsonl',header=header,data={},records=[fact(record_id='new')])
    save_artifact(tmp_path/'06_verify/verified_records.jsonl',header=header,data={},records=[fact(record_id='old')])
    export_knowledge(tmp_path)
    rows=[json.loads(s) for s in (tmp_path/'08_publish/knowledge/records.jsonl').read_text().splitlines()]
    assert [r['record_id'] for r in rows] == ['new']
    assert rows[0]['verification'] == 'UNVERIFIED'


def test_html_merged_headers_and_wider_rows_do_not_drop_columns() -> None:
    from tcm_extract_engine.core.tables import parse_table
    table = parse_table('<table><tr><th rowspan="2">对象</th><th colspan="2">剂量</th></tr><tr><th>下限</th><th>上限</th></tr><tr><td>A</td><td>1</td><td>3</td></tr></table>')
    assert table.headers == ['对象', '剂量 / 下限', '剂量 / 上限']
    assert table.rows == [['A', '1', '3']]
    assert table.cell_origins[0] == [(2, 0), (2, 1), (2, 2)]
    wide = parse_table('|A|\n|---|\n|1|2|')
    assert wide.rows == [['1', '2']]
    assert wide.warnings


@pytest.mark.asyncio
async def test_candidate_usage_is_delta_not_cumulative(tmp_path: Path) -> None:
    blocks, plan = write_case(tmp_path)
    shared = Client()
    first = await evaluate_candidate(book_id='b', blocks_path=blocks, plan_path=plan,
        target_template='knowledge.md', constraint_block='', extract_client=shared,
        verify_client=shared, work_dir=tmp_path/'c1')
    second = await evaluate_candidate(book_id='b', blocks_path=blocks, plan_path=plan,
        target_template='knowledge.md', constraint_block='', extract_client=shared,
        verify_client=shared, work_dir=tmp_path/'c2')
    assert first.llm_calls == second.llm_calls == 2


def test_migration_does_not_advance_unconfirmed_plan(tmp_path: Path) -> None:
    from tcm_extract_engine.core.manifest import BookManifest, BookStatus
    (tmp_path/'04_extract').mkdir()
    manifest = BookManifest(book_id='b', file_name='x.pdf', file_hash='h', status=BookStatus.WAITING_PLAN_CONFIRMATION)
    manifest.save(tmp_path)
    (tmp_path/'04_extract/records.jsonl').write_text(json.dumps({'record_id':'old','book_id':'b','record_type':'GENERAL_RULE','attributes':{}})+'\n')
    migrate_book(tmp_path, dry_run=False)
    loaded=BookManifest.load(tmp_path)
    assert loaded.status == BookStatus.WAITING_PLAN_CONFIRMATION
    assert loaded.derived_artifacts_stale
    loaded.save(tmp_path)
    assert BookManifest.load(tmp_path).derived_artifacts_stale
