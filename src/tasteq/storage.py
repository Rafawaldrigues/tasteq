"""SQLite persistence: tracks, sessions and every comparison ever made.

The comparison log is the source of truth. Models are never pickled; they are
rebuilt by replaying comparisons, which keeps the database readable, portable
across code versions and auditable.

Each comparison stores both the unit-cube points the model saw and the gains
that were actually played, the context used at the time, which version was
presented first and the excerpt, so later analysis (order effects, for
example) needs nothing else.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tasteq.analysis import TrackAnalysis
from tasteq.dsp import FloatArray
from tasteq.preference import Comparison, Outcome

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    fingerprint TEXT NOT NULL,
    duration_s REAL NOT NULL,
    loudness_lufs REAL NOT NULL,
    excerpt_start_s REAL NOT NULL,
    excerpt_duration_s REAL NOT NULL,
    features TEXT NOT NULL,
    analyzed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    model TEXT NOT NULL,
    config TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT
);
CREATE TABLE IF NOT EXISTS comparisons (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    track_id INTEGER REFERENCES tracks(id),
    created_at TEXT NOT NULL,
    subject TEXT NOT NULL,
    source TEXT NOT NULL,
    point_a TEXT NOT NULL,
    point_b TEXT NOT NULL,
    gains_a TEXT NOT NULL,
    gains_b TEXT NOT NULL,
    context TEXT NOT NULL,
    outcome TEXT NOT NULL,
    presented_first TEXT NOT NULL,
    excerpt_start_s REAL,
    excerpt_duration_s REAL,
    weight REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dump(array: FloatArray) -> str:
    return json.dumps([float(v) for v in np.asarray(array).ravel()])


def _load(text: str) -> FloatArray:
    return np.array(json.loads(text), dtype=np.float64)


@dataclass(frozen=True)
class StoredTrack:
    """A track row with its database id."""

    id: int
    analysis: TrackAnalysis


@dataclass(frozen=True)
class ComparisonRecord:
    """Everything logged about one comparison."""

    comparison: Comparison
    gains_a: FloatArray
    gains_b: FloatArray
    track_id: int | None
    presented_first: str
    source: str = "ab"
    subject: str = "human"
    excerpt_start_s: float | None = None
    excerpt_duration_s: float | None = None


class Storage:
    """Thin typed wrapper around one SQLite file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The demo serves requests from worker threads; it serializes access
        # with a lock, so sharing the connection across threads is safe.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        """Close the connection."""
        self.conn.close()

    def __enter__(self) -> Storage:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self.conn:
            yield self.conn

    # --- tracks ---

    def upsert_track(self, analysis: TrackAnalysis) -> int:
        """Insert or refresh a track's analysis; return its id."""
        with self._tx() as c:
            c.execute(
                """
                INSERT INTO tracks (path, fingerprint, duration_s, loudness_lufs,
                    excerpt_start_s, excerpt_duration_s, features, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    fingerprint = excluded.fingerprint,
                    duration_s = excluded.duration_s,
                    loudness_lufs = excluded.loudness_lufs,
                    excerpt_start_s = excluded.excerpt_start_s,
                    excerpt_duration_s = excluded.excerpt_duration_s,
                    features = excluded.features,
                    analyzed_at = excluded.analyzed_at
                """,
                (
                    analysis.path,
                    analysis.fingerprint,
                    analysis.duration_s,
                    analysis.loudness_lufs,
                    analysis.excerpt_start_s,
                    analysis.excerpt_duration_s,
                    json.dumps(analysis.features),
                    _now(),
                ),
            )
            row = c.execute("SELECT id FROM tracks WHERE path = ?", (analysis.path,)).fetchone()
        return int(row["id"])

    def fingerprint_of(self, path: str) -> str | None:
        """Return the stored fingerprint for ``path``, or ``None`` if never analyzed."""
        row = self.conn.execute("SELECT fingerprint FROM tracks WHERE path = ?", (path,)).fetchone()
        return None if row is None else str(row["fingerprint"])

    def tracks(self) -> list[StoredTrack]:
        """All analyzed tracks, by id."""
        rows = self.conn.execute("SELECT * FROM tracks ORDER BY id").fetchall()
        return [
            StoredTrack(
                id=int(r["id"]),
                analysis=TrackAnalysis(
                    path=r["path"],
                    fingerprint=r["fingerprint"],
                    duration_s=r["duration_s"],
                    loudness_lufs=r["loudness_lufs"],
                    excerpt_start_s=r["excerpt_start_s"],
                    excerpt_duration_s=r["excerpt_duration_s"],
                    features=tuple(json.loads(r["features"])),
                ),
            )
            for r in rows
        ]

    # --- sessions ---

    def start_session(self, kind: str, model: str, config: dict[str, Any]) -> int:
        """Open a session (calibrate, listen, blindtest) and return its id."""
        with self._tx() as c:
            cursor = c.execute(
                "INSERT INTO sessions (kind, model, config, started_at) VALUES (?, ?, ?, ?)",
                (kind, model, json.dumps(config), _now()),
            )
        return int(cursor.lastrowid or 0)

    def end_session(self, session_id: int) -> None:
        """Mark a session as finished."""
        with self._tx() as c:
            c.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (_now(), session_id))

    # --- comparisons ---

    def add_comparison(self, session_id: int, record: ComparisonRecord) -> int:
        """Append one comparison to the log."""
        comp = record.comparison
        with self._tx() as c:
            cursor = c.execute(
                """
                INSERT INTO comparisons (session_id, track_id, created_at, subject, source,
                    point_a, point_b, gains_a, gains_b, context, outcome, presented_first,
                    excerpt_start_s, excerpt_duration_s, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    record.track_id,
                    _now(),
                    record.subject,
                    record.source,
                    _dump(comp.a),
                    _dump(comp.b),
                    _dump(record.gains_a),
                    _dump(record.gains_b),
                    _dump(comp.context),
                    comp.outcome.value,
                    record.presented_first,
                    record.excerpt_start_s,
                    record.excerpt_duration_s,
                    comp.weight,
                ),
            )
        return int(cursor.lastrowid or 0)

    def comparisons(self, subject: str = "human") -> list[Comparison]:
        """Replayable history for one subject, oldest first."""
        rows = self.conn.execute(
            "SELECT * FROM comparisons WHERE subject = ? ORDER BY id", (subject,)
        ).fetchall()
        return [
            Comparison(
                a=_load(r["point_a"]),
                b=_load(r["point_b"]),
                context=_load(r["context"]),
                outcome=Outcome(r["outcome"]),
                weight=float(r["weight"]),
            )
            for r in rows
        ]

    def comparison_rows(self) -> list[dict[str, Any]]:
        """Raw comparison rows, for analysis and export."""
        return [dict(r) for r in self.conn.execute("SELECT * FROM comparisons ORDER BY id")]

    # --- meta ---

    def set_meta(self, key: str, value: Any) -> None:
        """Store a JSON value under ``key``."""
        with self._tx() as c:
            c.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        """Read a JSON value stored with :meth:`set_meta`."""
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])
