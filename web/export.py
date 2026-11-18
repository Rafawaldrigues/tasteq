"""Dump every participant's data to flat tables for analysis.

::

    python -m web.export --out exports/            # CSV
    python -m web.export --out exports/ --parquet  # also Parquet (needs pandas + pyarrow)

Writes four tables, all keyed by the anonymous participant ``code``:

- ``participants``: mode, device, styles, stage reached, status, counts,
  questionnaire and timestamps. ``status`` is ``completed``, ``withdrawn``,
  ``in_progress`` or ``abandoned`` (no activity for ``--stale-hours``), so
  partial runs can be analyzed too.
- ``comparisons``: every A/B answer that trained the model, with its phase
  (``session_kind``: screening or music), gains, context, outcome, and the
  timing and listening behaviour logged by the web front end.
- ``blind_trials``: every blind-test answer, with both conditions, whether the
  song was heard in phase 2 and the model size when the EQs were frozen.
- ``answers``: the raw ``web_answers`` log (one row per answer, all phases).

Databases are opened read-only, so exporting while the study runs is safe.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from web.config import Settings
from web.prepare import TRACKS_META_KEY
from web.study import STATE_KEY

Row = dict[str, Any]


def _meta(conn: sqlite3.Connection, key: str) -> Any:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else json.loads(row[0])


def _status(state: Row, stale_after: timedelta, now: datetime) -> str:
    if state["stage"] == "done":
        return "completed"
    if state["stage"] == "withdrawn":
        return "withdrawn"
    updated = datetime.fromisoformat(state["updated_at"])
    return "abandoned" if now - updated > stale_after else "in_progress"


def _answered(conn: sqlite3.Connection, stage: str) -> int:
    row = conn.execute("SELECT count(*) FROM web_answers WHERE stage = ?", (stage,)).fetchone()
    return int(row[0])


def read_participant(path: Path, stale_after: timedelta, now: datetime) -> dict[str, list[Row]]:
    """All rows of one participant database."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        state = _meta(conn, STATE_KEY)
        if state is None:
            return {}
        has_answers = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'web_answers'"
        ).fetchone()
        code = state["code"]
        tracks = {int(k): v for k, v in (_meta(conn, TRACKS_META_KEY) or {}).items()}
        questionnaire = state.get("questionnaire") or {}
        participant = {
            "code": code,
            "mode": state["mode"],
            "device": state["device"],
            "model_kind": state["model_kind"],
            "space": state["space"],
            "styles": ";".join(state.get("styles", [])),
            "stage": state["stage"],
            "withdrawn_at_stage": state.get("withdrawn_at_stage"),
            "status": _status(state, stale_after, now),
            "planned_screening": state["counts"]["screening"],
            "planned_music": state["counts"]["music"],
            "planned_blind": state["counts"]["blind"],
            "answered_screening": _answered(conn, "screening") if has_answers else 0,
            "answered_music": _answered(conn, "music") if has_answers else 0,
            "answered_blind": _answered(conn, "blind") if has_answers else 0,
            "music_tracks": ";".join(
                str(tracks[i]["catalog_id"]) for i in state.get("music_tracks", [])
            ),
            "holdout_tracks": ";".join(
                str(tracks[i]["catalog_id"]) for i in state.get("holdout_tracks", [])
            ),
            "phase1_gains": json.dumps(state.get("phase1_gains")),
            "phase1_n_comparisons": state.get("phase1_n_comparisons"),
            "learned_n_comparisons": state.get("learned_n_comparisons"),
            **{f"q_{k}": v for k, v in questionnaire.items()},
            "created_at": state["created_at"],
            "updated_at": state["updated_at"],
            "completed_at": state.get("completed_at"),
        }

        def track_fields(track_id: int | None) -> Row:
            info = tracks.get(track_id or -1, {})
            return {"track_kind": info.get("kind"), "catalog_id": info.get("catalog_id")}

        answers = (
            [dict(r) for r in conn.execute("SELECT * FROM web_answers ORDER BY id")]
            if has_answers
            else []
        )
        by_comparison = {a["comparison_id"]: a for a in answers if a["comparison_id"]}
        by_trial = {a["blind_trial_id"]: a for a in answers if a["blind_trial_id"]}
        timing = ("response_ms", "plays_first", "plays_second", "switches", "listen_ms")

        comparisons = []
        for r in conn.execute(
            "SELECT c.*, s.kind AS session_kind, s.model AS model FROM comparisons c "
            "JOIN sessions s ON s.id = c.session_id ORDER BY c.id"
        ):
            row = dict(r)
            answer = by_comparison.get(row["id"], {})
            comparisons.append(
                {
                    "code": code,
                    **row,
                    **track_fields(row["track_id"]),
                    "position": answer.get("position"),
                    **{k: answer.get(k) for k in timing},
                }
            )

        blind = []
        for r in conn.execute("SELECT * FROM blind_trials ORDER BY id"):
            row = dict(r)
            answer = by_trial.get(row["id"], {})
            detail = json.loads(answer.get("detail") or "{}")
            blind.append(
                {
                    "code": code,
                    **row,
                    **track_fields(row["track_id"]),
                    "heard_in_music": detail.get("heard_in_music"),
                    "position": answer.get("position"),
                    **{k: answer.get(k) for k in timing},
                }
            )
        return {
            "participants": [participant],
            "comparisons": comparisons,
            "blind_trials": blind,
            "answers": [{"code": code, **a, **track_fields(a["track_id"])} for a in answers],
        }
    finally:
        conn.close()


def collect(participants_dir: Path, stale_hours: float = 6.0) -> dict[str, list[Row]]:
    """Rows of every participant, concatenated per table."""
    now = datetime.now(UTC)
    tables: dict[str, list[Row]] = {
        "participants": [],
        "comparisons": [],
        "blind_trials": [],
        "answers": [],
    }
    for path in sorted(participants_dir.glob("*.sqlite")):
        for name, rows in read_participant(path, timedelta(hours=stale_hours), now).items():
            tables[name].extend(rows)
    return tables


def write_csv(rows: Iterable[Row], path: Path) -> None:
    """Write rows with the union of their keys as header."""
    rows = list(rows)
    columns: dict[str, None] = {}
    for row in rows:
        columns.update(dict.fromkeys(row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def export(
    participants_dir: Path, out: Path, parquet: bool = False, stale_hours: float = 6.0
) -> dict[str, int]:
    """Write every table to ``out``; return the row count per table."""
    out.mkdir(parents=True, exist_ok=True)
    tables = collect(participants_dir, stale_hours)
    for name, rows in tables.items():
        write_csv(rows, out / f"{name}.csv")
    if parquet:
        try:
            import pandas as pd

            for name, rows in tables.items():
                pd.DataFrame(rows).to_parquet(out / f"{name}.parquet", index=False)
        except ImportError as error:
            raise SystemExit(
                f"Parquet needs pandas and pyarrow ({error}); CSVs were written."
            ) from error
    return {name: len(rows) for name, rows in tables.items()}


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("exports"))
    parser.add_argument("--parquet", action="store_true", help="also write Parquet files")
    parser.add_argument("--stale-hours", type=float, default=6.0)
    args = parser.parse_args()
    counts = export(Settings.from_env().participants_dir, args.out, args.parquet, args.stale_hours)
    for name, n in counts.items():
        print(f"{name}: {n} rows")


if __name__ == "__main__":
    main()
