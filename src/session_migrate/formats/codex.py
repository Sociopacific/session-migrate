"""Codex rollout JSONL reader and conservative native writer."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections import Counter, deque
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from session_migrate.errors import SessionMigrateError
from session_migrate.formats.common import content_text, object_value, string, valid_rfc3339
from session_migrate.jsonl import JsonlRecord, encode_jsonl, file_sha256, iter_jsonl
from session_migrate.model import AgentFormat, Event, EventKind, Provenance, Role, Session

PINNED_CODEX_VERSION = "0.144.4"

# Codex 0.147+ paginated rollouts persist canonical TurnItems in
# event_msg/item_completed records. Provider response_item messages are not an
# equivalent transcript: they also contain synthetic environment and developer
# context. Paginated parsing therefore takes user/assistant messages only from
# the canonical completed items while retaining non-message response items for
# tool and reasoning data.
SUPPORTED_HISTORY_MODES = frozenset({"legacy", "paginated"})


def parse(path: Path) -> Session:
    records, history_base_resolved = _records_with_history_base(path)
    history_mode = _history_mode(records, history_base_resolved=history_base_resolved)
    if history_mode == "paginated":
        _validate_paginated_root(records)
    events: list[Event] = []
    fallback_events: list[Event] = []
    context_compacted_events: list[Event] = []
    session_id = None
    cwd = None
    started_at = None
    cli_version = None
    model = None
    model_provider = None
    response_message_count = 0
    response_messages: Counter[tuple[Role | None, str]] = Counter()
    title = None

    for record in records:
        value = record.value
        record_type = string(value.get("type"))
        timestamp = string(value.get("timestamp"))
        payload = object_value(value.get("payload"))
        provenance = Provenance(record.index, record_type)
        if record_type == "session_meta":
            session_id = (
                session_id or string(payload.get("id")) or string(payload.get("session_id"))
            )
            cwd_value = string(payload.get("cwd"))
            cwd = cwd or (Path(cwd_value) if cwd_value else None)
            started_at = started_at or string(payload.get("timestamp")) or timestamp
            cli_version = cli_version or string(payload.get("cli_version"))
            model_provider = model_provider or string(payload.get("model_provider"))
            continue
        if record_type == "response_item":
            if history_mode == "paginated" and string(payload.get("type")) in {
                "message",
                "agent_message",
            }:
                parsed = [
                    Event(
                        kind=EventKind.OPAQUE,
                        timestamp=timestamp,
                        payload={
                            "source_item_type": string(payload.get("type")) or "<missing>",
                            "reason": "paginated_provider_message",
                        },
                        provenance=provenance,
                    )
                ]
            else:
                parsed = _response_item_events(payload, timestamp, provenance)
            events.extend(parsed)
            response_message_count += sum(
                event.kind == EventKind.MESSAGE and event.role in {Role.USER, Role.ASSISTANT}
                for event in parsed
            )
            response_messages.update(
                _message_fingerprint(event)
                for event in parsed
                if event.kind == EventKind.MESSAGE
                and event.role in {Role.USER, Role.ASSISTANT}
                and event.text
            )
        elif record_type == "event_msg":
            event_type = string(payload.get("type"))
            if event_type == "item_completed" and history_mode == "paginated":
                events.extend(_paginated_completed_item_events(payload, timestamp, provenance))
            elif event_type == "user_message" and history_mode == "legacy":
                fallback_events.append(
                    Event(
                        kind=EventKind.MESSAGE,
                        role=Role.USER,
                        text=string(payload.get("message")),
                        timestamp=timestamp,
                        provenance=provenance,
                    )
                )
            elif event_type == "agent_message" and history_mode == "legacy":
                fallback_events.append(
                    Event(
                        kind=EventKind.MESSAGE,
                        role=Role.ASSISTANT,
                        text=string(payload.get("message")),
                        timestamp=timestamp,
                        provenance=provenance,
                    )
                )
            elif event_type == "task_started":
                model = model or string(payload.get("model"))
            elif event_type == "thread_name_updated":
                title = string(payload.get("thread_name")) or string(payload.get("name")) or title
            elif event_type == "context_compacted":
                context_compacted_events.append(
                    Event(
                        kind=EventKind.OPAQUE,
                        timestamp=timestamp,
                        payload={"source_event_type": event_type},
                        provenance=provenance,
                    )
                )
            elif event_type not in {"user_message", "agent_message"}:
                events.append(
                    Event(
                        kind=EventKind.OPAQUE,
                        timestamp=timestamp,
                        payload={"source_event_type": event_type or "<missing>"},
                        provenance=provenance,
                    )
                )
        elif record_type == "compacted":
            replacement_history = payload.get("replacement_history")
            if replacement_history is not None and not isinstance(replacement_history, list):
                raise SessionMigrateError("Codex replacement_history must be an array")
            events.append(
                Event(
                    kind=EventKind.COMPACTION,
                    role=Role.SYSTEM,
                    text=string(payload.get("message"))
                    or _replacement_history_summary(replacement_history),
                    timestamp=timestamp,
                    payload={
                        **(
                            {"replacement_history_expanded": True}
                            if replacement_history is not None
                            else {}
                        )
                    },
                    provenance=provenance,
                )
            )
        elif record_type == "turn_context":
            model = model or string(payload.get("model"))
            events.append(
                Event(
                    kind=EventKind.CONTEXT,
                    role=Role.SYSTEM,
                    timestamp=timestamp,
                    payload={"source_record_type": "turn_context"},
                    provenance=provenance,
                )
            )
        elif record_type in {"world_state", "security_risk_score"}:
            events.append(
                Event(
                    kind=EventKind.OPAQUE,
                    timestamp=timestamp,
                    payload={"source_record_type": record_type},
                    provenance=provenance,
                )
            )
        else:
            events.append(
                Event(
                    kind=EventKind.OPAQUE,
                    timestamp=timestamp,
                    payload={"source_record_type": record_type or "<missing>"},
                    provenance=provenance,
                )
            )

    if history_mode == "paginated":
        events.sort(key=lambda event: event.provenance.record_index)
    elif response_message_count == 0:
        events.extend(event for event in fallback_events if event.text)
        events.sort(key=lambda event: event.provenance.record_index)
    else:
        for event in fallback_events:
            fingerprint = _message_fingerprint(event)
            if event.text and response_messages[fingerprint]:
                response_messages[fingerprint] -= 1
                continue
            events.append(
                Event(
                    kind=EventKind.MESSAGE,
                    role=event.role,
                    text=event.text,
                    timestamp=event.timestamp,
                    payload={"ui_only_projection": True},
                    provenance=event.provenance,
                )
            )
        events.sort(key=lambda event: event.provenance.record_index)
    compaction_count = sum(event.kind == EventKind.COMPACTION for event in events)
    events.extend(context_compacted_events[compaction_count:])
    events.sort(key=lambda event: event.provenance.record_index)
    return Session(
        source_format=AgentFormat.CODEX,
        source_path=path.resolve(),
        source_sha256=file_sha256(path),
        session_id=session_id,
        cwd=cwd,
        started_at=started_at,
        cli_version=cli_version,
        model=model,
        title=title or (thread_name(path, session_id) if session_id else None),
        events=tuple(events),
        raw_record_count=len(records),
        model_provider=model_provider,
    )


def thread_name(path: Path, thread_id: str) -> str | None:
    """Thread name Codex keeps only in its state database (no thread_name_updated event).

    Codex Desktop names threads in ``state_<n>.sqlite`` ``threads.name``; the
    rollout itself often carries no title record. Read-only and best effort.
    """

    home = codex_home_for_rollout(path)
    if home is None:
        return None
    databases = sorted(
        home.glob("state_*.sqlite"),
        key=lambda candidate: (
            int(candidate.stem.split("_")[-1]) if candidate.stem.split("_")[-1].isdigit() else -1
        ),
    )
    if not databases:
        return catalog_titles(home).get(thread_id)
    # mode=ro needs the WAL side files; when Codex is closed they may be gone,
    # so fall back to an immutable snapshot read.
    for flags in ("mode=ro", "mode=ro&immutable=1"):
        with suppress(sqlite3.Error, OSError):
            connection = sqlite3.connect(f"{databases[-1].resolve().as_uri()}?{flags}", uri=True)
            try:
                row = connection.execute(
                    "SELECT name FROM threads WHERE id = ?", (thread_id,)
                ).fetchone()
            finally:
                connection.close()
            name = row[0].strip() if row and isinstance(row[0], str) else ""
            return name or catalog_titles(home).get(thread_id)
    return catalog_titles(home).get(thread_id)


def catalog_titles(home: Path) -> dict[str, str]:
    """Titles Codex Desktop shows in its sidebar (``sqlite/codex-dev.db`` thread catalog).

    The sidebar is built from this catalog rather than ``state_<n>.sqlite``; it
    also titles threads that never got a ``threads.name``.
    """

    database = home / "sqlite" / "codex-dev.db"
    if not database.is_file():
        return {}
    for flags in ("mode=ro", "mode=ro&immutable=1"):
        with suppress(sqlite3.Error, OSError):
            connection = sqlite3.connect(f"{database.resolve().as_uri()}?{flags}", uri=True)
            try:
                rows = connection.execute(
                    "SELECT thread_id, display_title FROM local_thread_catalog"
                ).fetchall()
            finally:
                connection.close()
            return {
                thread_id: title.strip()
                for thread_id, title in rows
                if isinstance(thread_id, str) and isinstance(title, str) and title.strip()
            }
    return {}


def _replacement_history_summary(history: Any) -> str | None:
    """Readable context Codex kept after a server-side (encrypted) compaction.

    Newer Codex compactions carry no summary text: the model continues from an
    encrypted compaction item plus the user messages listed in
    ``replacement_history``. Those messages are the readable part of that context.
    """

    if not isinstance(history, list):
        return None
    lines: list[str] = []
    for item in history:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        role = string(item.get("role"))
        if role not in {"user", "assistant"}:
            continue
        content = item.get("content")
        blocks = content if isinstance(content, list) else [{"text": content}]
        for block in blocks:
            text = string(block.get("text")) if isinstance(block, dict) else None
            if not text or text.lstrip().startswith("<"):
                continue
            lines.append(f"[{role}] {text.strip()}")
    if not lines:
        return None
    return (
        "Codex compacted this conversation on its server; its summary is encrypted. "
        "Messages Codex kept in context after the compaction:\n\n" + "\n\n".join(lines)
    )


def _history_mode(records: list[Any], *, history_base_resolved: bool = False) -> str:
    selected_mode = "legacy"
    metadata_seen = False
    for record in records:
        if string(record.value.get("type")) != "session_meta":
            continue
        payload = object_value(record.value.get("payload"))
        history_mode = string(payload.get("history_mode")) or "legacy"
        if history_mode not in SUPPORTED_HISTORY_MODES:
            raise SessionMigrateError(
                f"Codex history mode {history_mode!r} is not supported; "
                f"expected one of {', '.join(sorted(SUPPORTED_HISTORY_MODES))}"
            )
        if payload.get("history_base") is not None and not history_base_resolved:
            raise SessionMigrateError("Codex history_base lineage is not supported")
        if payload.get("subagent_history_start_ordinal") is not None:
            raise SessionMigrateError(
                "Codex paginated subagent history projection is not supported"
            )
        if metadata_seen and history_mode != selected_mode:
            raise SessionMigrateError("Codex session metadata has conflicting history modes")
        selected_mode = history_mode
        metadata_seen = True
    if selected_mode == "paginated" and (
        not records or string(records[0].value.get("type")) != "session_meta"
    ):
        raise SessionMigrateError("Codex paginated history must start with session metadata")
    return selected_mode


def _records_with_history_base(
    path: Path, *, _visited: frozenset[Path] = frozenset()
) -> tuple[list[JsonlRecord], bool]:
    """Read a rollout, prepending its history_base prefix when Codex split the thread.

    Codex continues a long paginated thread in a new rollout file whose
    session_meta carries ``history_base`` (same thread id plus the exclusive end
    ordinal of the prefix kept in an earlier file). The earlier file may itself
    be a continuation, so the lineage is resolved recursively.
    """

    resolved_path = path.resolve()
    if resolved_path in _visited:
        raise SessionMigrateError("Codex history_base lineage contains a cycle")
    records = list(iter_jsonl(path))
    base = _history_base(records)
    if base is None:
        return records, False
    thread_id, end_ordinal = base
    base_path = locate_history_base(resolved_path, thread_id, end_ordinal)
    base_records, _ = _records_with_history_base(base_path, _visited=_visited | {resolved_path})
    # A continuation may restart a few ordinals before the declared end of its
    # base (seen in Codex 0.154 branch segments); its own copy of that overlap
    # is the one on the visible branch, so the base is cut where it starts.
    first = _ordinal(records[0]) if records else -1
    cut = first if 0 < first < end_ordinal else end_ordinal
    prefix = [record for record in base_records if _ordinal(record) < cut]
    if not prefix or _ordinal(prefix[-1]) != cut - 1:
        raise SessionMigrateError(
            "Codex history_base prefix is incomplete; "
            f"expected ordinals below {cut} in the base rollout"
        )
    combined = [*prefix, *records]
    return [
        JsonlRecord(index=index, line_number=record.line_number, value=record.value)
        for index, record in enumerate(combined)
    ], True


def _history_base(records: list[JsonlRecord]) -> tuple[str, int] | None:
    for record in records:
        if string(record.value.get("type")) != "session_meta":
            continue
        base = record.value.get("payload", {}).get("history_base")
        if base is None:
            return None
        if not isinstance(base, dict):
            raise SessionMigrateError("Codex history_base must be an object")
        thread_id = string(base.get("thread_id"))
        end_ordinal = base.get("end_ordinal_exclusive")
        if (
            not thread_id
            or isinstance(end_ordinal, bool)
            or not isinstance(end_ordinal, int)
            or end_ordinal <= 0
        ):
            raise SessionMigrateError("Codex history_base is missing thread_id or end ordinal")
        return thread_id, end_ordinal
    return None


def codex_home_for_rollout(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name in {"sessions", "archived_sessions"}:
            return parent.parent
    return None


def thread_rollouts(home: Path, thread_id: str) -> list[Path]:
    """Every rollout file of one thread: the original and its continuations."""

    # Continuation files are named ``rollout-<ts>-<root id>_<segment id>.jsonl``
    # and a later segment may point at an earlier one by its segment id.
    patterns = (
        f"rollout-*-{thread_id}.jsonl",
        f"rollout-*-{thread_id}_*.jsonl",
        f"rollout-*_{thread_id}.jsonl",
    )
    found: list[Path] = []
    for pattern in patterns:
        found.extend(home.glob(f"sessions/*/*/*/{pattern}"))
        found.extend((home / "archived_sessions").glob(pattern))
    return sorted({candidate.resolve() for candidate in found})


def first_ordinal(path: Path) -> int | None:
    with suppress(OSError, ValueError, StopIteration):
        with path.open("rb") as stream:
            value = json.loads(stream.readline())
        ordinal = value.get("ordinal") if isinstance(value, dict) else None
        if isinstance(ordinal, int) and not isinstance(ordinal, bool):
            return ordinal
    return None


def locate_history_base(path: Path, thread_id: str, end_ordinal: int) -> Path:
    home = codex_home_for_rollout(path)
    candidates = thread_rollouts(home, thread_id) if home is not None else []
    if not candidates:
        candidates = sorted(
            {
                *path.parent.glob(f"rollout-*-{thread_id}.jsonl"),
                *path.parent.glob(f"rollout-*-{thread_id}_*.jsonl"),
                *path.parent.glob(f"rollout-*_{thread_id}.jsonl"),
            }
        )
    starts = [
        (start, candidate)
        for candidate in candidates
        if candidate.resolve() != path
        and (start := first_ordinal(candidate)) is not None
        and start < end_ordinal
    ]
    if not starts:
        raise SessionMigrateError(
            f"Codex history_base rollout for thread {thread_id} was not found"
        )
    return max(starts)[1]


def _ordinal(record: JsonlRecord) -> int:
    ordinal = record.value.get("ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int):
        return -1
    return ordinal


def _validate_paginated_root(records: list[Any]) -> None:
    """Fail closed if a root rollout is not a complete canonical ordinal stream.

    One gap is accepted: right after the session metadata. Codex writes it when
    a thread starts from the middle of a history (an external import that keeps
    only the tail of a long session, or a repaired thread); the rest of the
    stream must still be contiguous.
    """

    offset = 0
    for index, record in enumerate(records):
        ordinal = record.value.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise SessionMigrateError(
                f"Codex paginated record {record.index} is missing an integer ordinal"
            )
        if index == 1 and ordinal > 1 and string(records[0].value.get("type")) == "session_meta":
            offset = ordinal - 1
        expected = index + offset
        if ordinal != expected:
            raise SessionMigrateError(
                "Codex paginated ordinals must be contiguous from zero; "
                f"record {record.index} has ordinal {ordinal}, expected {expected}"
            )


def _paginated_completed_item_events(
    payload: dict[str, Any],
    timestamp: str | None,
    provenance: Provenance,
) -> list[Event]:
    item = object_value(payload.get("item"))
    item_type = string(item.get("type"))
    if item_type == "UserMessage":
        return _paginated_user_message_events(item, timestamp, provenance)
    if item_type == "AgentMessage":
        content = item.get("content")
        if not isinstance(content, list):
            return [_opaque_completed_item(item_type, timestamp, provenance, "invalid_content")]
        result: list[Event] = []
        for block_index, block in enumerate(content):
            block_provenance = Provenance(
                provenance.record_index,
                provenance.record_type,
                block_index=block_index,
            )
            if not isinstance(block, dict):
                result.append(
                    _opaque_completed_item(item_type, timestamp, block_provenance, "invalid_block")
                )
                continue
            block_type = string(block.get("type"))
            text_value = string(block.get("text"))
            if block_type in {"Text", "text"} and text_value:
                result.append(
                    Event(
                        kind=EventKind.MESSAGE,
                        role=Role.ASSISTANT,
                        text=text_value,
                        timestamp=timestamp,
                        provenance=block_provenance,
                    )
                )
            else:
                result.append(
                    _opaque_completed_item(
                        item_type,
                        timestamp,
                        block_provenance,
                        f"unsupported_block:{block_type or '<missing>'}",
                    )
                )
        return result
    return [_opaque_completed_item(item_type, timestamp, provenance)]


def _paginated_user_message_events(
    item: dict[str, Any],
    timestamp: str | None,
    provenance: Provenance,
) -> list[Event]:
    content = item.get("content")
    if not isinstance(content, list):
        return [_opaque_completed_item("UserMessage", timestamp, provenance, "invalid_content")]
    result: list[Event] = []
    for block_index, block in enumerate(content):
        block_provenance = Provenance(
            provenance.record_index,
            provenance.record_type,
            block_index=block_index,
        )
        if not isinstance(block, dict):
            result.append(
                _opaque_completed_item("UserMessage", timestamp, block_provenance, "invalid_block")
            )
            continue
        block_type = string(block.get("type"))
        if block_type == "text":
            text_value = string(block.get("text"))
            if text_value:
                result.append(
                    Event(
                        kind=EventKind.MESSAGE,
                        role=Role.USER,
                        text=text_value,
                        timestamp=timestamp,
                        provenance=block_provenance,
                    )
                )
        elif block_type == "image":
            result.append(
                Event(
                    kind=EventKind.CONTEXT,
                    role=Role.USER,
                    timestamp=timestamp,
                    payload={
                        "block_type": "image",
                        "image_url": string(block.get("image_url")),
                    },
                    provenance=block_provenance,
                )
            )
        elif block_type == "audio":
            result.append(
                Event(
                    kind=EventKind.CONTEXT,
                    role=Role.USER,
                    timestamp=timestamp,
                    payload={
                        "block_type": "audio",
                        "audio_url": string(block.get("audio_url")),
                    },
                    provenance=block_provenance,
                )
            )
        else:
            result.append(
                _opaque_completed_item(
                    "UserMessage",
                    timestamp,
                    block_provenance,
                    f"unsupported_block:{block_type or '<missing>'}",
                )
            )
    return result


def _opaque_completed_item(
    item_type: str | None,
    timestamp: str | None,
    provenance: Provenance,
    reason: str | None = None,
) -> Event:
    return Event(
        kind=EventKind.OPAQUE,
        timestamp=timestamp,
        payload={
            "source_event_type": "item_completed",
            "source_item_type": item_type or "<missing>",
            **({"reason": reason} if reason else {}),
        },
        provenance=provenance,
    )


def serialize(
    session: Session,
    *,
    session_id: str,
    cwd: Path,
    cli_version: str = PINNED_CODEX_VERSION,
    model_provider: str = "openai",
    timestamp: str | None = None,
) -> tuple[bytes, dict[str, int]]:
    """Serialize the model-visible subset accepted by Codex CLI 0.144.4."""

    fallback_timestamp = valid_rfc3339(timestamp) or valid_rfc3339(session.started_at) or _utc_now()
    records: list[dict[str, Any]] = [
        {
            "timestamp": fallback_timestamp,
            "type": "session_meta",
            "payload": {
                "session_id": session_id,
                "id": session_id,
                "timestamp": fallback_timestamp,
                "cwd": str(cwd),
                "originator": "session-migrate",
                "cli_version": cli_version,
                "source": "cli",
                "model_provider": model_provider,
                "history_mode": "legacy",
            },
        }
    ]
    dropped: Counter[str] = Counter()
    generated_tool_ids: deque[str] = deque()
    seen_tool_call_ids: set[str] = set()
    seen_tool_result_ids: set[str] = set()
    for event in session.events:
        event_timestamp = valid_rfc3339(event.timestamp)
        if event.timestamp and not event_timestamp:
            dropped["timestamp:invalid"] += 1
        event_timestamp = event_timestamp or fallback_timestamp
        if (
            event.kind == EventKind.MESSAGE
            and event.text
            and event.role in {Role.USER, Role.ASSISTANT}
        ):
            if event.payload.get("ui_only_projection") is True:
                dropped["message:ui_only_projection"] += 1
            if event.role == Role.ASSISTANT:
                records.append(
                    _envelope(
                        event_timestamp,
                        "event_msg",
                        {"type": "agent_message", "message": event.text},
                    )
                )
                content_type = "output_text"
                role = "assistant"
            else:
                records.append(
                    _envelope(
                        event_timestamp,
                        "event_msg",
                        {"type": "user_message", "message": event.text},
                    )
                )
                content_type = "input_text"
                role = "user"
            records.append(
                _envelope(
                    event_timestamp,
                    "response_item",
                    {
                        "type": "message",
                        "role": role,
                        "content": [{"type": content_type, "text": event.text}],
                    },
                )
            )
        elif event.kind == EventKind.TOOL_CALL:
            call_id = event.tool_call_id
            if not call_id:
                call_id = f"call_session_migrate_{uuid.uuid4().hex}"
                generated_tool_ids.append(call_id)
                dropped["tool_call:missing_id"] += 1
            tool_name = event.tool_name
            if not tool_name:
                tool_name = "unknown_tool"
                dropped["tool_call:missing_name"] += 1
            arguments = event.payload.get("input", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            if call_id in seen_tool_call_ids:
                dropped["tool_call:duplicate_id"] += 1
            seen_tool_call_ids.add(call_id)
            records.append(
                _envelope(
                    event_timestamp,
                    "response_item",
                    {
                        "type": "function_call",
                        "name": tool_name,
                        "arguments": arguments,
                        "call_id": call_id,
                    },
                )
            )
            if event.payload.get("namespace"):
                dropped["tool_call:namespace"] += 1
        elif event.kind == EventKind.TOOL_RESULT:
            source_call_id = event.tool_call_id
            call_id = source_call_id
            if not call_id:
                call_id = (
                    generated_tool_ids.popleft()
                    if generated_tool_ids
                    else f"call_missing_{uuid.uuid4().hex}"
                )
                dropped["tool_result:missing_id"] += 1
            elif call_id not in seen_tool_call_ids:
                dropped["tool_result:orphan_id"] += 1
            if source_call_id and source_call_id in seen_tool_result_ids:
                dropped["tool_result:duplicate_id"] += 1
            if source_call_id:
                seen_tool_result_ids.add(source_call_id)
            output, omitted_blocks = _codex_tool_result_output(event)
            dropped.update(omitted_blocks)
            records.append(
                _envelope(
                    event_timestamp,
                    "response_item",
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output,
                    },
                )
            )
            if event.payload.get("is_error") is True:
                dropped["tool_result:is_error"] += 1
        elif (
            event.kind == EventKind.CONTEXT
            and event.payload.get("block_type") == "image"
            and event.role == Role.USER
        ):
            image_url = string(event.payload.get("image_url"))
            if image_url:
                records.append(
                    _envelope(
                        event_timestamp,
                        "response_item",
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_image", "image_url": image_url}],
                        },
                    )
                )
            else:
                dropped["context:image"] += 1
        elif event.kind == EventKind.COMPACTION and event.text:
            records.append(
                _envelope(
                    event_timestamp,
                    "compacted",
                    {"message": event.text},
                )
            )
            if event.payload.get("has_boundary_metadata") is True:
                dropped["compaction:boundary_metadata"] += 1
        else:
            dropped[_omission_key(event)] += 1
    if session.title:
        dropped["session:title"] += 1
    return encode_jsonl(records), dict(sorted(dropped.items()))


def rollout_relative_path(session_id: str, timestamp: str) -> Path:
    date = _parse_date(timestamp)
    filename_timestamp = date.strftime("%Y-%m-%dT%H-%M-%S")
    return (
        Path("sessions")
        / date.strftime("%Y")
        / date.strftime("%m")
        / date.strftime("%d")
        / f"rollout-{filename_timestamp}-{session_id}.jsonl"
    )


def _response_item_events(
    payload: dict[str, Any],
    timestamp: str | None,
    provenance: Provenance,
) -> list[Event]:
    item_type = string(payload.get("type"))
    if item_type == "message":
        role_name = string(payload.get("role"))
        if role_name == "assistant":
            role = Role.ASSISTANT
        elif role_name == "user":
            role = Role.USER
        elif role_name in {"developer", "system"}:
            role = Role.SYSTEM
        else:
            return [
                Event(
                    kind=EventKind.OPAQUE,
                    timestamp=timestamp,
                    payload={"reason": "unknown_message_role"},
                    provenance=provenance,
                )
            ]
        result: list[Event] = []
        content = payload.get("content")
        if not isinstance(content, list):
            return result
        for block_index, block in enumerate(content):
            block_provenance = Provenance(
                provenance.record_index,
                provenance.record_type,
                block_index=block_index,
            )
            if not isinstance(block, dict):
                result.append(Event(kind=EventKind.OPAQUE, provenance=block_provenance))
                continue
            block_type = string(block.get("type"))
            if block_type in {"input_text", "output_text", "text"}:
                text = string(block.get("text"))
                if text:
                    result.append(
                        Event(
                            kind=EventKind.MESSAGE,
                            role=role,
                            text=text,
                            timestamp=timestamp,
                            provenance=block_provenance,
                        )
                    )
            elif block_type in {"input_image", "image"}:
                image_url = string(block.get("image_url")) or string(block.get("url"))
                result.append(
                    Event(
                        kind=EventKind.CONTEXT,
                        role=role,
                        timestamp=timestamp,
                        payload={"block_type": "image", "image_url": image_url},
                        provenance=block_provenance,
                    )
                )
            else:
                result.append(
                    Event(
                        kind=EventKind.OPAQUE,
                        role=role,
                        timestamp=timestamp,
                        payload={"source_block_type": block_type or "<missing>"},
                        provenance=block_provenance,
                    )
                )
        return result
    if item_type in {"function_call", "custom_tool_call"}:
        arguments: Any = payload.get("arguments", payload.get("input", {}))
        if isinstance(arguments, str):
            with suppress(json.JSONDecodeError):
                arguments = json.loads(arguments)
        return [
            Event(
                kind=EventKind.TOOL_CALL,
                role=Role.ASSISTANT,
                timestamp=timestamp,
                tool_name=string(payload.get("name")),
                tool_call_id=string(payload.get("call_id")) or string(payload.get("id")),
                payload={
                    "input": arguments,
                    **(
                        {"namespace": payload["namespace"]}
                        if string(payload.get("namespace"))
                        else {}
                    ),
                },
                provenance=provenance,
            )
        ]
    if item_type in {"function_call_output", "custom_tool_call_output"}:
        normalized_content = _portable_tool_result_content(payload.get("output"))
        return [
            Event(
                kind=EventKind.TOOL_RESULT,
                role=Role.TOOL,
                timestamp=timestamp,
                text=content_text(payload.get("output")),
                tool_call_id=string(payload.get("call_id")),
                payload={"content_blocks": normalized_content},
                provenance=provenance,
            )
        ]
    if item_type == "reasoning":
        return [
            Event(
                kind=EventKind.THINKING,
                role=Role.ASSISTANT,
                timestamp=timestamp,
                payload={"source_item_type": "reasoning"},
                provenance=provenance,
            )
        ]
    if item_type:
        return [
            Event(
                kind=EventKind.OPAQUE,
                timestamp=timestamp,
                payload={"source_item_type": item_type},
                provenance=provenance,
            )
        ]
    return []


def _portable_tool_result_content(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if not isinstance(value, list):
        return [{"type": "opaque"}]
    result: list[dict[str, Any]] = []
    for block in value:
        if not isinstance(block, dict):
            result.append({"type": "opaque"})
            continue
        block_type = string(block.get("type"))
        if block_type in {"input_text", "text"}:
            text = string(block.get("text"))
            if text:
                result.append({"type": "text", "text": text})
        elif block_type in {"input_image", "image"}:
            image_url = string(block.get("image_url")) or string(block.get("url"))
            if image_url:
                result.append({"type": "image", "image_url": image_url})
            else:
                result.append({"type": "opaque"})
        elif block_type == "tool_reference":
            tool_name = string(block.get("tool_name"))
            if tool_name:
                result.append({"type": "tool_reference", "tool_name": tool_name})
            else:
                result.append({"type": "opaque"})
        elif block_type == "input_audio":
            result.append({"type": "audio"})
        else:
            result.append({"type": "opaque"})
    return result


def _codex_tool_result_output(event: Event) -> tuple[str | list[dict[str, Any]], Counter[str]]:
    blocks = event.payload.get("content_blocks")
    if not isinstance(blocks, list):
        return event.text or "", Counter()
    result: list[dict[str, Any]] = []
    omitted: Counter[str] = Counter()
    for portable in blocks:
        if not isinstance(portable, dict):
            omitted["tool_result:block"] += 1
            continue
        block_type = portable.get("type")
        if block_type == "text" and isinstance(portable.get("text"), str):
            result.append({"type": "input_text", "text": portable["text"]})
        elif block_type == "image" and isinstance(portable.get("image_url"), str):
            result.append({"type": "input_image", "image_url": portable["image_url"]})
        else:
            omitted[f"tool_result:{block_type or 'block'}"] += 1
    if not result:
        return event.text or "", omitted
    if len(result) == 1 and result[0].get("type") == "input_text":
        return str(result[0]["text"]), omitted
    return result, omitted


def _envelope(timestamp: str, record_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"timestamp": timestamp, "type": record_type, "payload": payload}


def _message_fingerprint(event: Event) -> tuple[Role | None, str]:
    normalized = (event.text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return event.role, normalized


def _omission_key(event: Event) -> str:
    if event.kind == EventKind.MESSAGE and event.role == Role.SYSTEM:
        return "message:privileged_role"
    if event.kind == EventKind.COMPACTION and event.payload.get("replacement_history_expanded"):
        return "compaction:replacement_history_expanded"
    if event.kind == EventKind.CONTEXT:
        if event.role == Role.SYSTEM and event.payload.get("block_type") == "image":
            return "context:privileged_image"
        if event.payload.get("source_record_type"):
            return f"context:{event.payload['source_record_type']}"
        return f"context:{event.payload.get('block_type', 'unknown')}"
    if event.kind == EventKind.OPAQUE:
        detail = next(
            (
                event.payload.get(key)
                for key in (
                    "reason",
                    "source_record_type",
                    "source_event_type",
                    "source_block_type",
                    "source_item_type",
                )
                if event.payload.get(key)
            ),
            "unknown",
        )
        return f"opaque:{detail}"
    return event.kind.value


def _parse_date(timestamp: str) -> datetime:
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return datetime.now(UTC)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
