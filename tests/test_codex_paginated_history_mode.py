"""Codex 0.147+ writes history_mode="paginated"; earlier releases wrote "legacy".

The paginated layout keeps the model-visible items this adapter reads at the same
top level. It adds `item_completed` wrappers around items that are already
emitted as their own records, plus `token_usage_record` accounting — neither
carries model-visible content. These tests pin that equivalence so the adapter
cannot start depending on the wrappers, and pin that a forked session
(`history_base`) is still refused rather than silently truncated.
"""

from pathlib import Path

import pytest

from session_migrate.errors import SessionMigrateError
from session_migrate.formats import codex

FIXTURES = Path(__file__).parent / "fixtures"
LEGACY = FIXTURES / "codex-0.144.4" / "basic.jsonl"
PAGINATED = FIXTURES / "codex-0.153.4" / "paginated.jsonl"


def _visible(session):
    """Role/text pairs a target format receives, excluding inert records.

    Paginated sessions carry `item_completed` and `token_usage_record` entries
    that the adapter keeps as OPAQUE events — preserved for provenance, never
    rendered into a target transcript. They are excluded here so this compares
    the conversation itself.
    """
    return [(event.role, event.text) for event in session.events if event.role is not None]


def test_paginated_history_mode_parses():
    session = codex.parse(PAGINATED)
    assert session.cli_version == "0.153.4"
    assert session.events, "paginated session produced no events"


def test_paginated_matches_legacy_conversation():
    """The added wrappers must not change, duplicate, or drop visible turns."""
    assert _visible(codex.parse(PAGINATED)) == _visible(codex.parse(LEGACY))


def test_paginated_extra_records_stay_opaque():
    """`item_completed` duplicates items already parsed; it must not become a turn.

    This is the real hazard in accepting the paginated layout: `item_completed`
    wraps items that are ALSO emitted as their own `response_item` records, so
    interpreting both would double every assistant message.
    """
    session = codex.parse(PAGINATED)
    extra = [event for event in session.events if event.role is None]
    assert extra, "expected the paginated-only records to be retained"
    assert {event.payload.get("source_record_type") for event in extra} == {
        "item_completed",
        "token_usage_record",
    }
    assert all(event.kind.value == "opaque" for event in extra)


def test_unknown_history_mode_still_refused(tmp_path):
    """An unrecognized mode must fail closed, not be parsed hopefully."""
    lines = PAGINATED.read_text().splitlines()
    lines[0] = lines[0].replace('"paginated"', '"some-future-mode"')
    path = tmp_path / "future.jsonl"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(SessionMigrateError, match="history mode"):
        codex.parse(path)


def test_history_base_still_refused(tmp_path):
    """A fork's earlier turns live in another file; refusing beats truncating."""
    import json

    lines = PAGINATED.read_text().splitlines()
    meta = json.loads(lines[0])
    meta["payload"]["history_base"] = {
        "thread_id": "10000000-0000-4000-8000-000000000000",
        "end_ordinal_exclusive": 12,
    }
    lines[0] = json.dumps(meta)
    path = tmp_path / "forked.jsonl"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(SessionMigrateError, match="history_base"):
        codex.parse(path)
