"""Codex 0.147+ root paginated-history compatibility and safety tests."""

import json
from pathlib import Path

import pytest

from session_migrate.conversion import ConversionOptions, convert_session
from session_migrate.errors import SessionMigrateError
from session_migrate.formats import codex
from session_migrate.model import EventKind, Role, TargetFormat

FIXTURES = Path(__file__).parent / "fixtures"
LEGACY = FIXTURES / "codex-0.144.4" / "basic.jsonl"
PAGINATED = FIXTURES / "codex-0.153.4" / "paginated.jsonl"


def _portable(session):
    return [
        (
            event.kind,
            event.role,
            event.text,
            event.tool_name,
            event.tool_call_id,
            event.payload.get("block_type"),
            event.payload.get("image_url"),
            event.payload.get("content_blocks"),
        )
        for event in session.events
        if event.kind != EventKind.OPAQUE
    ]


def _rewrite_fixture(tmp_path: Path, mutate) -> Path:
    records = [json.loads(line) for line in PAGINATED.read_text().splitlines()]
    mutate(records)
    path = tmp_path / "paginated.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_paginated_uses_canonical_completed_items_and_matches_legacy() -> None:
    session = codex.parse(PAGINATED)

    assert session.cli_version == "0.153.4"
    assert _portable(session) == _portable(codex.parse(LEGACY))
    assert "INTERNAL ENVIRONMENT CONTEXT" not in " ".join(
        event.text or "" for event in session.events
    )
    assert [
        (event.role, event.text) for event in session.events if event.kind == EventKind.MESSAGE
    ] == [
        (Role.USER, "Remember synthetic migrator nonce BETA-2048."),
        (Role.ASSISTANT, "I will remember the synthetic nonce."),
        (Role.USER, "Continue after the synthetic compaction."),
        (Role.ASSISTANT, "The synthetic post-compaction fixture is complete."),
    ]


def test_paginated_provider_messages_and_accounting_stay_opaque() -> None:
    session = codex.parse(PAGINATED)
    opaque = [event for event in session.events if event.kind == EventKind.OPAQUE]

    assert sum(event.payload.get("reason") == "paginated_provider_message" for event in opaque) == 5
    assert (
        sum(event.payload.get("source_record_type") == "token_usage_record" for event in opaque)
        == 1
    )


@pytest.mark.parametrize("target_format", tuple(TargetFormat))
def test_paginated_context_cannot_leak_to_any_target(
    tmp_path: Path, target_format: TargetFormat
) -> None:
    artifact = convert_session(
        codex.parse(PAGINATED),
        ConversionOptions(
            target_format=target_format,
            session_id="50000000-0000-4000-8000-000000000001",
            cwd=tmp_path,
        ),
    )

    assert artifact.native_bytes
    assert b"INTERNAL ENVIRONMENT CONTEXT" not in artifact.native_bytes


@pytest.mark.parametrize("ordinal", [None, True, "3", 99])
def test_paginated_invalid_or_noncontiguous_ordinals_fail_closed(
    tmp_path: Path, ordinal: object
) -> None:
    def mutate(records):
        if ordinal is None:
            records[3].pop("ordinal")
        else:
            records[3]["ordinal"] = ordinal

    path = _rewrite_fixture(tmp_path, mutate)
    with pytest.raises(SessionMigrateError, match="ordinal"):
        codex.parse(path)


def test_unknown_history_mode_still_refused(tmp_path: Path) -> None:
    def mutate(records):
        records[0]["payload"]["history_mode"] = "some-future-mode"

    path = _rewrite_fixture(tmp_path, mutate)
    with pytest.raises(SessionMigrateError, match="history mode"):
        codex.parse(path)


def test_history_base_still_refused(tmp_path: Path) -> None:
    def mutate(records):
        records[0]["payload"]["history_base"] = {
            "thread_id": "10000000-0000-4000-8000-000000000000",
            "end_ordinal_exclusive": 12,
            "end_byte_offset": 1024,
        }

    path = _rewrite_fixture(tmp_path, mutate)
    with pytest.raises(SessionMigrateError, match="history_base"):
        codex.parse(path)


def test_paginated_subagent_projection_still_refused(tmp_path: Path) -> None:
    def mutate(records):
        records[0]["payload"]["subagent_history_start_ordinal"] = 8

    path = _rewrite_fixture(tmp_path, mutate)
    with pytest.raises(SessionMigrateError, match="subagent history projection"):
        codex.parse(path)


def test_paginated_metadata_must_be_first_and_consistent(tmp_path: Path) -> None:
    def move_meta(records):
        records[0], records[1] = records[1], records[0]
        records[0]["ordinal"], records[1]["ordinal"] = 0, 1

    with pytest.raises(SessionMigrateError, match="start with session metadata"):
        codex.parse(_rewrite_fixture(tmp_path, move_meta))

    def duplicate_conflicting_meta(records):
        duplicate = json.loads(json.dumps(records[0]))
        duplicate["payload"]["history_mode"] = "legacy"
        records.append(duplicate)
        for ordinal, record in enumerate(records):
            record["ordinal"] = ordinal

    with pytest.raises(SessionMigrateError, match="conflicting history modes"):
        codex.parse(_rewrite_fixture(tmp_path, duplicate_conflicting_meta))
