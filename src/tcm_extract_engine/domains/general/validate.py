import json

from tcm_extract_engine.core.records import SourceRecord


def validate_record(record: SourceRecord) -> list:
    return []  # Core schema validation and source checks apply to every general fact.


def dedup_key(record: SourceRecord) -> tuple:
    return (
        record.record_type,
        json.dumps(record.payload, sort_keys=True, ensure_ascii=False),
        tuple(record.source.block_ids),
    )


def coverage_member_count(records: list) -> list:
    return []


def coverage_table_rows(records: list, blocks: list) -> list:
    return []
