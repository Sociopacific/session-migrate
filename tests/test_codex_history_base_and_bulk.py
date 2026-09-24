"""Codex history_base continuation files and bulk Codex -> Claude import."""

import json
from pathlib import Path

from session_migrate.bulk import bulk_codex_to_claude
from session_migrate.discovery import locate_session
from session_migrate.formats import codex
from session_migrate.model import AgentFormat, EventKind

FIXTURES = Path(__file__).parent / "fixtures"
PAGINATED = FIXTURES / "codex-0.153.4" / "paginated.jsonl"
THREAD_ID = "01a07899-73a0-7cb2-8999-2af3623e2221"
SEGMENT_ID = "01a08550-b54e-7371-943f-628e1ca6b7ad"


def _records() -> list[dict]:
    records = [json.loads(line) for line in PAGINATED.read_text().splitlines()]
    for record in records:
        if record["type"] == "session_meta":
            record["payload"]["id"] = THREAD_ID
            record["payload"]["session_id"] = THREAD_ID
    return records


def _write(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def _split_thread(home: Path, split: int) -> tuple[Path, Path]:
    """Write the fixture as Codex does after rolling a long thread into a new file."""

    records = _records()
    day = home / "sessions" / "2026" / "09" / "09"
    base = _write(day / f"rollout-2026-09-09T11-37-34-{THREAD_ID}.jsonl", records[:split])
    meta = json.loads(json.dumps(records[0]))
    meta["ordinal"] = split
    meta["payload"]["history_base"] = {
        "thread_id": THREAD_ID,
        "end_ordinal_exclusive": split,
        "end_byte_offset": 1,
    }
    tail = []
    for record in records[split:]:
        record = json.loads(json.dumps(record))
        record["ordinal"] += 1
        tail.append(record)
    continuation = _write(
        day / f"rollout-2026-09-09T12-00-00-{THREAD_ID}_{SEGMENT_ID}.jsonl", [meta, *tail]
    )
    return base, continuation


def _messages(session) -> list[tuple]:
    return [(event.role, event.text) for event in session.events if event.kind == EventKind.MESSAGE]


def test_continuation_prepends_history_base_prefix(tmp_path: Path) -> None:
    split = len(_records()) // 2
    _, continuation = _split_thread(tmp_path / "codex", split)

    session = codex.parse(continuation)

    assert session.session_id == THREAD_ID
    assert _messages(session) == _messages(codex.parse(PAGINATED))


def test_discovery_picks_the_latest_continuation(tmp_path: Path) -> None:
    home = tmp_path / "codex"
    _, continuation = _split_thread(home, len(_records()) // 2)

    assert locate_session(AgentFormat.CODEX, THREAD_ID, home) == continuation.resolve()


def test_bulk_imports_leaf_once_and_skips_subagents_and_external_imports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "codex"
    target = tmp_path / "claude"
    _split_thread(source, len(_records()) // 2)

    subagent = _records()
    subagent[0]["payload"]["id"] = "01a07899-0000-7000-8000-000000000001"
    subagent[0]["payload"]["source"] = {"subagent": {"other": "guardian"}}
    _write(source / "sessions/2026/09/10/rollout-2026-09-10T00-00-00-sub.jsonl", subagent)

    imported = _records()
    imported[0]["payload"]["id"] = "01a07899-0000-7000-8000-000000000002"
    imported[1].setdefault("payload", {})["turn_id"] = "external-import-turn-1"
    _write(source / "sessions/2026/09/10/rollout-2026-09-10T00-00-01-imp.jsonl", imported)

    first = bulk_codex_to_claude(source_home=source, target_home=target)
    assert len(first.migrated) == 1
    assert first.failed == []
    assert first.skipped == {
        "continued_in_newer_rollout": 1,
        "subagent": 1,
        "imported_into_codex_from_other_agent": 1,
    }
    assert Path(first.migrated[0]["output"]).is_file()

    second = bulk_codex_to_claude(source_home=source, target_home=target)
    assert second.migrated == []
    assert second.skipped["already_migrated"] == 1


def test_bulk_dry_run_writes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "codex"
    target = tmp_path / "claude"
    _split_thread(source, len(_records()) // 2)

    report = bulk_codex_to_claude(source_home=source, target_home=target, dry_run=True)

    assert len(report.migrated) == 1
    assert not target.exists()


def test_title_falls_back_to_codex_state_database_name(tmp_path: Path) -> None:
    import sqlite3

    home = tmp_path / "codex"
    _, continuation = _split_thread(home, len(_records()) // 2)
    database = sqlite3.connect(home / "state_5.sqlite")
    database.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, name TEXT)")
    database.execute("INSERT INTO threads VALUES (?, ?)", (THREAD_ID, "Named in Codex Desktop"))
    database.commit()
    database.close()

    assert codex.parse(continuation).title == "Named in Codex Desktop"
