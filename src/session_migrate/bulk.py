"""Bulk migration of every Codex rollout into Claude Code sessions."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import uuid
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from session_migrate.conversion import (
    ConversionOptions,
    convert_session,
    load_session,
    target_import_paths,
    write_artifact,
)
from session_migrate.errors import SessionMigrateError
from session_migrate.formats import codex
from session_migrate.model import AgentFormat, TargetFormat

# Stable namespace so re-running bulk maps each source thread to the same
# target session ID and skips what was already installed.
_TASK_KEY = re.compile(r"([A-Z][A-Z0-9]*-\d+)-")
BULK_NAMESPACE = uuid.UUID("5f0d3a52-8c1e-4b8e-9a51-6d2b8f7e4c10")


@dataclass
class BulkReport:
    scanned: int = 0
    migrated: list[dict[str, Any]] = field(default_factory=list)
    already_migrated_ids: list[str] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)
    failed: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self, *, dry_run: bool) -> dict[str, Any]:
        return {
            "dry_run": dry_run,
            "scanned": self.scanned,
            "migrated_count": len(self.migrated),
            "failed_count": len(self.failed),
            "skipped": dict(sorted(self.skipped.items())),
            "migrated": self.migrated,
            "failed": self.failed,
        }


@dataclass(frozen=True)
class _Rollout:
    path: Path
    meta: dict[str, Any]
    external_import: bool

    @property
    def thread_id(self) -> str:
        return str(self.meta.get("id") or self.meta.get("session_id") or "")

    @property
    def segment_id(self) -> str:
        stem = self.path.stem
        return stem.rsplit("_", 1)[1] if "_" in stem else self.thread_id


def bulk_codex_to_claude(
    *,
    source_home: Path,
    target_home: Path,
    include_subagents: bool = False,
    include_archived: bool = False,
    since: date | None = None,
    only_named: bool = False,
    archived_task_cli_state: Path | None = None,
    dry_run: bool = False,
) -> BulkReport:
    report = BulkReport()
    rollouts = _scan_rollouts(source_home, include_archived=include_archived)
    report.scanned = len(rollouts)

    superseded: set[Path] = set()
    for rollout in rollouts:
        base = rollout.meta.get("history_base")
        if not isinstance(base, dict):
            continue
        with suppress(SessionMigrateError, ValueError, TypeError):
            superseded.add(
                codex.locate_history_base(
                    rollout.path,
                    str(base.get("thread_id") or ""),
                    int(base.get("end_ordinal_exclusive") or 0),
                )
            )

    archived_task_keys = (
        _archived_task_cli_keys(archived_task_cli_state) if archived_task_cli_state else set()
    )
    external_sources = _external_import_sources(source_home)
    selected: list[_Rollout] = []
    for rollout in rollouts:
        if rollout.path in superseded:
            report.skipped["continued_in_newer_rollout"] += 1
        elif rollout.external_import and not _import_diverged(
            rollout, external_sources.get(rollout.thread_id)
        ):
            # Codex's own import of another agent's session (for example
            # Claude Code) that was never continued in Codex; the original
            # still exists at its source.
            report.skipped["imported_into_codex_from_other_agent"] += 1
        elif not include_subagents and _is_subagent(rollout.meta):
            report.skipped["subagent"] += 1
        elif since is not None and _started_on(rollout.meta) < since:
            report.skipped["before_since"] += 1
        elif archived_task_keys and _task_key_in_path(
            str(rollout.meta.get("cwd") or ""), archived_task_keys
        ):
            report.skipped["archived_task_cli_environment"] += 1
        elif only_named and not codex.thread_name(rollout.path, rollout.thread_id):
            report.skipped["unnamed"] += 1
        else:
            selected.append(rollout)

    thread_counts = Counter(rollout.thread_id for rollout in selected)
    previously_migrated = _migrated_codex_threads(target_home)
    for rollout in selected:
        earlier = previously_migrated.get(rollout.thread_id)
        if earlier is not None and thread_counts[rollout.thread_id] == 1:
            # Imported before, possibly by `transfer` under a random target ID.
            report.skipped["already_migrated"] += 1
            report.already_migrated_ids.append(earlier)
            continue
        key = (
            rollout.thread_id
            if thread_counts[rollout.thread_id] == 1
            else f"{rollout.thread_id}:{rollout.segment_id}"
        )
        target_id = str(uuid.uuid5(BULK_NAMESPACE, f"codex:{key}"))
        try:
            session = load_session(rollout.path, AgentFormat.CODEX)
            artifact = convert_session(
                session,
                ConversionOptions(target_format=TargetFormat.CLAUDE, session_id=target_id),
            )
            output_path, manifest_path = target_import_paths(artifact, target_home)
            if output_path.exists() or manifest_path.exists():
                report.skipped["already_migrated"] += 1
                report.already_migrated_ids.append(artifact.session_id)
                continue
            if not dry_run:
                write_artifact(artifact, output_path=output_path, manifest_path=manifest_path)
        except SessionMigrateError as exc:
            report.failed.append({"source": str(rollout.path), "error": str(exc)})
            continue
        report.migrated.append(
            {
                "source_id": rollout.thread_id,
                "session_id": artifact.session_id,
                "cwd": str(artifact.cwd),
                "title": session.title,
                "output": str(output_path),
            }
        )
    return report


def _archived_task_cli_keys(state_path: Path) -> set[str]:
    """Task keys whose every Task CLI environment is archived."""

    try:
        tasks = json.loads(state_path.read_text()).get("tasks", {})
    except (OSError, ValueError, AttributeError):
        return set()
    archived: dict[str, bool] = {}
    for name, task in tasks.items() if isinstance(tasks, dict) else ():
        match = _TASK_KEY.match(name)
        if match and isinstance(task, dict):
            key = match.group(1)
            archived[key] = archived.get(key, True) and bool(task.get("archived"))
    return {key for key, is_archived in archived.items() if is_archived}


def _task_key_in_path(cwd: str, keys: set[str]) -> bool:
    for part in Path(cwd).parts:
        match = _TASK_KEY.match(part)
        if match and match.group(1) in keys:
            return True
    return False


def _migrated_codex_threads(target_home: Path) -> dict[str, str]:
    """Codex thread ID -> Claude session ID from existing session-migrate manifests."""

    migrated: dict[str, str] = {}
    for manifest in (target_home / "session-migrate" / "manifests").glob("*.json"):
        with suppress(OSError, ValueError):
            data = json.loads(manifest.read_text())
            source = data.get("source") if isinstance(data, dict) else None
            if isinstance(source, dict) and source.get("format") == AgentFormat.CODEX.value:
                thread_id = source.get("session_id")
                if isinstance(thread_id, str) and thread_id:
                    migrated.setdefault(thread_id, manifest.stem)
    return migrated


def _scan_rollouts(home: Path, *, include_archived: bool) -> list[_Rollout]:
    paths = list(home.glob("sessions/*/*/*/rollout-*.jsonl"))
    if include_archived:
        paths.extend((home / "archived_sessions").glob("rollout-*.jsonl"))
    rollouts: list[_Rollout] = []
    for path in sorted(paths):
        try:
            with path.open("rb") as stream:
                first = json.loads(stream.readline())
                second_line = stream.readline()
            second = json.loads(second_line) if second_line.strip() else {}
        except (OSError, ValueError):
            continue
        if not isinstance(first, dict) or first.get("type") != "session_meta":
            continue
        payload = first.get("payload")
        if not isinstance(payload, dict):
            continue
        rollouts.append(
            _Rollout(path=path.resolve(), meta=payload, external_import=_is_external_import(second))
        )
    return rollouts


def _external_import_sources(home: Path) -> dict[str, Path]:
    """Codex thread ID -> source transcript it was imported from (Codex /import log)."""

    try:
        records = json.loads((home / "external_agent_session_imports.json").read_text())
    except (OSError, ValueError):
        return {}
    sources: dict[str, Path] = {}
    for record in records.get("records", []) if isinstance(records, dict) else []:
        if not isinstance(record, dict):
            continue
        thread_id, source = record.get("imported_thread_id"), record.get("source_path")
        if isinstance(thread_id, str) and isinstance(source, str):
            sources[thread_id] = Path(source)
    return sources


def _import_diverged(rollout: _Rollout, source: Path | None) -> bool:
    """True when an imported thread holds work its source does not have."""

    if source is not None and not source.exists():
        return True
    try:
        with rollout.path.open("rb") as stream:
            for line in stream:
                if b'"task_started"' in line and b"external-import-" not in line:
                    return True
    except OSError:
        return False
    return False


def _is_external_import(record: Any) -> bool:
    payload = record.get("payload") if isinstance(record, dict) else None
    turn_id = payload.get("turn_id") if isinstance(payload, dict) else None
    return isinstance(turn_id, str) and turn_id.startswith("external-import-")


def _is_subagent(meta: dict[str, Any]) -> bool:
    source = meta.get("source")
    return meta.get("thread_source") == "subagent" or (
        isinstance(source, dict) and "subagent" in source
    )


def _started_on(meta: dict[str, Any]) -> date:
    timestamp = str(meta.get("timestamp") or "")
    try:
        return date.fromisoformat(timestamp[:10])
    except ValueError:
        return date.min


def claude_desktop_sessions_root() -> Path:
    return Path.home() / "Library" / "Application Support" / "Claude" / "claude-code-sessions"


def registered_in_claude_desktop(session_id: str, root: Path | None = None) -> bool:
    root = root or claude_desktop_sessions_root()
    return any(root.glob(f"*/*/local_{session_id}.json"))


def register_in_claude_desktop(session_ids: list[str], *, delay: float = 1.5) -> list[str]:
    """Hand each Claude Code session to Claude Desktop through its resume deep link.

    Desktop's ``claude://resume?session=<id>`` handler imports a CLI transcript
    from ~/.claude/projects into its own session list. Sessions already in the
    Desktop store are skipped. Returns the IDs that were handed over.
    """

    if sys.platform != "darwin":
        raise SessionMigrateError("Claude Desktop registration is supported on macOS only")
    handed: list[str] = []
    for session_id in session_ids:
        if registered_in_claude_desktop(session_id):
            continue
        subprocess.run(["open", f"claude://resume?session={session_id}"], check=True)
        handed.append(session_id)
        time.sleep(delay)
    return handed
