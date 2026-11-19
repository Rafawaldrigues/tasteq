"""One participant's run through the study, and the registry that holds them.

Every participant gets their own SQLite file, copied from the pre-analyzed
template, so participants can never see or disturb each other's model or
log, and a participant's data can be deleted by removing one file. The
tasteq tables (sessions, comparisons, blind_trials) are filled by
:class:`tasteq.session.Tuner` exactly as in the CLI; the study adds its
stage machine (stored as JSON in ``meta``) and a ``web_answers`` table with
timings and listening behaviour for every answer.

Stages, in order::

    volume -> screening -> styles -> music -> blind -> questionnaire -> done

A participant can also leave at any point (``withdrawn``); whatever they
answered so far stays in their file unless they ask for it to be deleted.
"""

from __future__ import annotations

import json
import math
import re
import secrets
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tasteq.analysis import diverse_subset
from tasteq.blindtest import Trial, analyze, matched_random
from tasteq.dsp import DEFAULT_CENTER_FREQS_HZ, FloatArray, frequency_response_db
from tasteq.library import Library
from tasteq.session import Answer, Question, Tuner
from tasteq.space import EQSpace
from tasteq.storage import Storage, StoredTrack
from web.audio import ClipCache, ClipRenderer, encode
from web.config import Settings
from web.prepare import TRACKS_META_KEY

STATE_KEY = "web_state"
QUESTION_STAGES = ("screening", "music", "blind")
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I
CODE_PATTERN = re.compile(r"^[A-Z2-9]{4}-[A-Z2-9]{4}$")
CURVE_FREQS_HZ = np.geomspace(20.0, 20_000.0, 96)
IDLE_EVICT_S = 600.0

ANSWERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS web_answers (
    id INTEGER PRIMARY KEY,
    stage TEXT NOT NULL,
    position INTEGER NOT NULL,
    track_id INTEGER,
    comparison_id INTEGER,
    blind_trial_id INTEGER,
    answer TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    answered_at TEXT NOT NULL,
    response_ms INTEGER NOT NULL,
    plays_first INTEGER,
    plays_second INTEGER,
    switches INTEGER,
    listen_ms INTEGER,
    detail TEXT NOT NULL DEFAULT '{}'
);
"""


class StudyError(Exception):
    """A request that does not fit the participant's current stage."""

    status_code = 400


class StaleQuestionError(StudyError):
    """The answer is for a question that is no longer open (reload, double click)."""

    status_code = 409


class UnknownParticipantError(StudyError):
    """No participant with that code."""

    status_code = 404


def now_iso() -> str:
    """UTC timestamp with milliseconds."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def new_code() -> str:
    """Random anonymous participant code such as ``K7QF-2MZD`` (40 bits)."""
    chars = [secrets.choice(CODE_ALPHABET) for _ in range(8)]
    return "".join(chars[:4]) + "-" + "".join(chars[4:])


def curve(gains: FloatArray) -> list[float]:
    """EQ magnitude response at :data:`CURVE_FREQS_HZ`, rounded for JSON."""
    return [round(float(v), 2) for v in frequency_response_db(gains, CURVE_FREQS_HZ)]


def _gains(values: Any) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


def _listify(gains: FloatArray) -> list[float]:
    return [round(float(v), 4) for v in gains]


@dataclass
class OpenQuestion:
    """The A/B question a participant is listening to right now."""

    qid: str
    stage: str
    position: int
    track: StoredTrack
    issued_at: str
    issued_clock: float
    audio: tuple[bytes, bytes]
    media_type: str
    question: Question | None = None  # screening and music
    trial: Trial | None = None  # blind test


class Participant:
    """Stage machine, model and log of one participant. All public methods lock."""

    def __init__(self, settings: Settings, clips: ClipCache, path: Path) -> None:
        self.settings = settings
        self.clips = clips
        self.path = path
        self.lock = threading.RLock()
        self.storage = Storage(path)
        self.storage.conn.executescript(ANSWERS_SCHEMA)
        state = self.storage.get_meta(STATE_KEY)
        if state is None:
            raise UnknownParticipantError(f"{path.name} has no study state")
        self.state: dict[str, Any] = state
        self.track_info: dict[int, dict[str, Any]] = {
            int(k): v for k, v in self.storage.get_meta(TRACKS_META_KEY, {}).items()
        }
        self.open: OpenQuestion | None = None
        self.last_used = time.monotonic()
        self._tuner: Tuner | None = None

    # --- creation ---

    @classmethod
    def create(
        cls, settings: Settings, clips: ClipCache, code: str, mode: str, device: str
    ) -> Participant:
        """Copy the template for a new participant and open it."""
        study = settings.study
        if mode not in study.modes:
            raise StudyError(f"unknown mode {mode!r}")
        if device not in study.devices:
            raise StudyError(f"unknown device {device!r}")
        path = settings.participants_dir / f"{code}.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(settings.template_path, path)
        counts = study.modes[mode]
        state = {
            "code": code,
            "mode": mode,
            "device": device,
            "model_kind": study.model_kind,
            "space": study.space,
            "seed": secrets.randbelow(2**31),
            "counts": {"screening": counts.screening, "music": counts.music, "blind": counts.blind},
            "plan": {
                "questions_per_sound": study.screening.questions_per_sound,
                "questions_per_track": study.music.questions_per_track,
                "blind_pairs": [list(p) for p in study.blind_pairs],
            },
            "stage": "volume",
            "position": 0,
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }
        with Storage(path) as storage:
            storage.set_meta(STATE_KEY, state)
        return cls(settings, clips, path)

    # --- model ---

    @property
    def tuner(self) -> Tuner:
        """The participant's model, rebuilt from their log the first time it is needed."""
        if self._tuner is None:
            stage = self.state["stage"]
            kind = "music" if stage not in ("volume", "screening") else "screening"
            self._tuner = Tuner(
                self.storage,
                model_kind=self.state["model_kind"],
                space=EQSpace(kind=self.state["space"]),
                session_kind=kind,
                seed=int(self.state["seed"]),
                renderer=ClipRenderer(self.clips),
            )
            if stage not in ("volume", "screening", "music"):
                self._tuner.close()
        return self._tuner

    def _track(self, track_id: int) -> StoredTrack:
        return next(t for t in self.tuner.library.tracks if t.id == track_id)

    def _tracks_of_kind(self, kind: str) -> list[StoredTrack]:
        return [
            t
            for t in self.tuner.library.tracks
            if self.track_info.get(t.id, {}).get("kind") == kind
        ]

    def _rng(self, salt: int) -> np.random.Generator:
        return np.random.default_rng([int(self.state["seed"]), salt])

    # --- status ---

    @property
    def code(self) -> str:
        """Anonymous participant code."""
        return str(self.state["code"])

    def status(self) -> dict[str, Any]:
        """Where the participant is, for the front end."""
        with self.lock:
            return self._status()

    def _status(self) -> dict[str, Any]:
        stage = self.state["stage"]
        counts = self.state["counts"]
        reached_stage = self.state.get("withdrawn_at_stage", stage)
        order = ["volume", "screening", "styles", "music", "blind", "questionnaire", "done"]
        reached = order.index(reached_stage)
        answered = 0
        for name in QUESTION_STAGES:
            if order.index(name) < reached:
                answered += counts[name]
            elif name == reached_stage:
                answered += self.state["position"]
        return {
            "code": self.code,
            "mode": self.state["mode"],
            "stage": stage,
            "position": self.state["position"],
            "stage_total": counts.get(stage, 0),
            "answered": answered,
            "total": sum(counts.values()),
            "styles": self.state.get("styles", []),
        }

    def _save(self) -> None:
        self.state["updated_at"] = now_iso()
        self.storage.set_meta(STATE_KEY, self.state)

    def _require(self, *stages: str) -> None:
        if self.state["stage"] not in stages:
            raise StudyError(f"not possible in stage {self.state['stage']!r}")

    # --- simple stage steps ---

    def volume_set(self) -> dict[str, Any]:
        """Start phase 1 once the participant has set a comfortable volume."""
        with self.lock:
            self._require("volume")
            sounds = self._tracks_of_kind("synthetic")
            per = self.state["plan"]["questions_per_sound"]
            n_sounds = min(len(sounds), math.ceil(self.state["counts"]["screening"] / per))
            contexts = np.array([self.tuner.library.context(t) for t in sounds])
            chosen = [sounds[i].id for i in diverse_subset(contexts, n_sounds)]
            self._rng(1).shuffle(chosen)
            self.state.update(stage="screening", position=0, screening_tracks=chosen)
            self.state["volume_set_at"] = now_iso()
            self._save()
            return self._status()

    def choose_styles(self, styles: list[str]) -> dict[str, Any]:
        """Pick the songs for phase 2 (and those held out for the blind test)."""
        with self.lock:
            self._require("styles")
            catalog_styles = self.settings.catalog.styles
            styles = list(dict.fromkeys(styles))
            minimum = self.settings.study.music.min_styles
            if len(styles) < minimum or any(s not in catalog_styles for s in styles):
                raise StudyError(f"choose at least {minimum} of {sorted(catalog_styles)}")
            style_of = {t.id: t.style for t in self.settings.catalog.tracks}
            pool = [
                t
                for t in self._tracks_of_kind("music")
                if style_of.get(self.track_info[t.id]["catalog_id"]) in styles
            ]
            if not pool:
                raise StudyError("no songs available for these styles")
            self.state["styles"] = styles
            self._plan_music(pool)
            # What the model predicts for each chosen song from phase 1 alone
            # (no music answers yet). The blind test's "phase1" condition plays
            # it, so learned-vs-phase1 measures what training on the songs added.
            chosen = self.state["music_tracks"] + self.state["holdout_tracks"]
            self.state["phase1_track_gains"] = {
                str(i): _listify(self.tuner.best_gains(self._track(i))) for i in chosen
            }
            self.tuner.session_id = self.storage.start_session(
                "music", self.tuner.model.name, {"styles": styles}
            )
            self.state.update(stage="music", position=0)
            self._save()
            return self._status()

    def _plan_music(self, pool: list[StoredTrack]) -> None:
        cfg = self.settings.study.music
        rng = self._rng(2)
        per = self.state["plan"]["questions_per_track"]
        wanted = max(cfg.min_tracks, math.ceil(self.state["counts"]["music"] / per))
        n_holdout = max(0, min(cfg.holdout_tracks, len(pool) - cfg.min_tracks))
        order = rng.permutation(len(pool))
        holdout = [pool[i] for i in order[:n_holdout]]
        rest = [pool[i] for i in order[n_holdout:]]
        contexts = np.array([self.tuner.library.context(t) for t in rest])
        music = [rest[i] for i in diverse_subset(contexts, min(wanted, len(rest)))]
        self.state["music_tracks"] = [t.id for t in music]
        self.state["holdout_tracks"] = [t.id for t in holdout]

    def submit_questionnaire(self, answers: dict[str, Any]) -> dict[str, Any]:
        """Store the final questionnaire; the study is complete."""
        with self.lock:
            self._require("questionnaire")
            self.state["questionnaire"] = answers
            self.state.update(stage="done", position=0, completed_at=now_iso())
            self._save()
            return self._status()

    def withdraw(self) -> None:
        """Leave the study, keeping what was answered so far."""
        with self.lock:
            if self.state["stage"] not in ("done", "withdrawn"):
                self.state["withdrawn_at_stage"] = self.state["stage"]
                self.state.update(stage="withdrawn", withdrawn_at=now_iso())
                self._save()

    # --- questions ---

    def question(self) -> dict[str, Any]:
        """Return the open question of the current stage, opening one if needed."""
        with self.lock:
            self._require(*QUESTION_STAGES)
            current = self.open
            if (
                current is None
                or current.stage != self.state["stage"]
                or current.position != self.state["position"]
            ):
                current = self.open = self._open_question()
            return self._question_payload(current)

    def audio(self, qid: str, slot: int) -> tuple[bytes, str]:
        """Return the encoded audio of version 1 or 2 of the open question."""
        with self.lock:
            if self.open is None or self.open.qid != qid or slot not in (1, 2):
                raise StaleQuestionError("this question is no longer open")
            return self.open.audio[slot - 1], self.open.media_type

    def _schedule_track(self, key: str, per: int) -> StoredTrack:
        tracks = self.state[key]
        return self._track(tracks[(self.state["position"] // per) % len(tracks)])

    def _open_question(self) -> OpenQuestion:
        stage = self.state["stage"]
        tuner = self.tuner
        question = trial = None
        if stage == "blind":
            item = self.state["blind_plan"][self.state["position"]]
            track = self._track(item["track_id"])
            trial = Trial(
                track,
                (item["conditions"][0], item["conditions"][1]),
                (_gains(item["gains"][0]), _gains(item["gains"][1])),
                bool(item["a_first"]),
            )
            first, second = trial.presented
            audio = (tuner.renderer.excerpt(track, first), tuner.renderer.excerpt(track, second))
        else:
            if stage == "screening":
                track = self._schedule_track(
                    "screening_tracks", self.state["plan"]["questions_per_sound"]
                )
            else:
                track = self._schedule_track(
                    "music_tracks", self.state["plan"]["questions_per_track"]
                )
            question = tuner.ask(track)
            audio = tuner.render_question(question)
        fmt = self.settings.study.audio.format
        rate = tuner.renderer.sample_rate
        first_bytes, media_type = encode(audio[0], rate, fmt)
        second_bytes, _ = encode(audio[1], rate, fmt)
        return OpenQuestion(
            qid=secrets.token_urlsafe(8),
            stage=stage,
            position=self.state["position"],
            track=track,
            issued_at=now_iso(),
            issued_clock=time.monotonic(),
            audio=(first_bytes, second_bytes),
            media_type=media_type,
            question=question,
            trial=trial,
        )

    def _question_payload(self, current: OpenQuestion) -> dict[str, Any]:
        info = self.track_info.get(current.track.id, {})
        return {
            "qid": current.qid,
            "stage": current.stage,
            "position": current.position,
            "stage_total": self.state["counts"][current.stage],
            "track": {
                "kind": info.get("kind"),
                "name": info.get("name"),
                "catalog_id": info.get("catalog_id"),
            },
            "audio": [f"/api/p/{self.code}/audio/{current.qid}/{slot}" for slot in (1, 2)],
            "status": self._status(),
        }

    def answer(
        self, qid: str, answer: Answer, listening: dict[str, int] | None = None
    ) -> dict[str, Any]:
        """Record the answer to the open question and move on."""
        with self.lock:
            current = self.open
            if current is None or current.qid != qid or current.stage != self.state["stage"]:
                raise StaleQuestionError("this question is no longer open")
            comparison_id = blind_id = None
            detail: dict[str, Any] = {}
            if current.question is not None:
                self.tuner.answer(current.question, answer)
                row = self.storage.conn.execute("SELECT max(id) FROM comparisons").fetchone()
                comparison_id = int(row[0])
            elif current.trial is not None:
                trial = current.trial
                blind_id = self.storage.add_blind_trial(
                    int(self.state["blind_session_id"]),
                    trial.track.id,
                    trial.conditions,
                    trial.gains,
                    presented_first="a" if trial.a_first else "b",
                    outcome=trial.outcome(answer),
                    n_comparisons=self.tuner.n_comparisons,
                )
                detail["heard_in_music"] = trial.track.id in self.state["music_tracks"]
            self._log_answer(current, answer, comparison_id, blind_id, listening or {}, detail)
            self.open = None
            self.state["position"] += 1
            if self.state["position"] >= self.state["counts"][current.stage]:
                self._finish_stage(current.stage)
            self._save()
            return self._status()

    def _log_answer(
        self,
        current: OpenQuestion,
        answer: str,
        comparison_id: int | None,
        blind_id: int | None,
        listening: dict[str, int],
        detail: dict[str, Any],
    ) -> None:
        with self.storage.conn as c:
            c.execute(
                """
                INSERT INTO web_answers (stage, position, track_id, comparison_id,
                    blind_trial_id, answer, issued_at, answered_at, response_ms,
                    plays_first, plays_second, switches, listen_ms, detail)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    current.stage,
                    current.position,
                    current.track.id,
                    comparison_id,
                    blind_id,
                    answer,
                    current.issued_at,
                    now_iso(),
                    int(1000 * (time.monotonic() - current.issued_clock)),
                    listening.get("plays_first"),
                    listening.get("plays_second"),
                    listening.get("switches"),
                    listening.get("listen_ms"),
                    json.dumps(detail),
                ),
            )

    def _finish_stage(self, stage: str) -> None:
        tuner = self.tuner
        if stage == "screening":
            self.state["phase1_n_comparisons"] = tuner.n_comparisons
            tuner.close()
            self.state.update(stage="styles", position=0)
        elif stage == "music":
            tuner.close()
            self._plan_blind()
            self.state.update(stage="blind", position=0)
        elif stage == "blind":
            self.storage.end_session(int(self.state["blind_session_id"]))
            self.state.update(stage="questionnaire", position=0)

    def _plan_blind(self) -> None:
        """Freeze every EQ the blind test will play; later answers cannot change them."""
        tuner = self.tuner
        rng = self._rng(3)
        n_trials = self.state["counts"]["blind"]
        pairs = [tuple(p) for p in self.state["plan"]["blind_pairs"]]
        # Held-out songs first so they are in the rotation even for short tests.
        tracks = [self._track(i) for i in self.state["holdout_tracks"] + self.state["music_tracks"]]
        learned = {t.id: tuner.best_gains(t) for t in tracks}
        limit = tuner.space.spec.max_gain_db
        phase1 = {int(k): _gains(v) for k, v in self.state["phase1_track_gains"].items()}
        plan = []
        for i in range(n_trials):
            track = tracks[i % len(tracks)]
            # Shift the pair each round so a track does not always meet the same pair.
            pair = pairs[(i + i // len(tracks)) % len(pairs)]
            gains = {
                "learned": learned[track.id],
                "flat": np.zeros_like(learned[track.id]),
                "phase1": phase1[track.id],
            }
            other = (
                matched_random(learned[track.id], rng, limit)
                if pair[1] == "random"
                else gains[pair[1]]
            )
            versions = [(pair[0], gains[pair[0]]), (pair[1], other)]
            if rng.random() < 0.5:  # which condition is "a" is random too
                versions.reverse()
            plan.append(
                {
                    "track_id": track.id,
                    "conditions": [versions[0][0], versions[1][0]],
                    "gains": [_listify(versions[0][1]), _listify(versions[1][1])],
                    "a_first": bool(rng.random() < 0.5),
                }
            )
        order = rng.permutation(len(plan))
        self.state["blind_plan"] = [plan[i] for i in order]
        self.state["learned_gains"] = {str(k): _listify(v) for k, v in learned.items()}
        self.state["learned_n_comparisons"] = tuner.n_comparisons
        self.state["blind_session_id"] = self.storage.start_session(
            "blindtest",
            tuner.model.name,
            {"n_trials": n_trials, "pairs": self.state["plan"]["blind_pairs"]},
        )

    # --- results ---

    def results(self) -> dict[str, Any]:
        """Learned curves per song and the blind-test score."""
        with self.lock:
            self._require("questionnaire", "done")
            pairs = [(p[0], p[1]) for p in self.state["plan"]["blind_pairs"]]
            rows = [
                r
                for r in self.storage.blind_trials()
                if r["session_id"] == self.state["blind_session_id"]
            ]
            blind = [
                {
                    "pair": list(r.pair),
                    "wins": r.wins,
                    "losses": r.losses,
                    "ties": r.ties,
                    "win_rate": None if r.n == 0 else round(r.win_rate, 3),
                    "p_value": round(r.p_value, 4),
                }
                for r in analyze(rows, pairs)
            ]
            phase1 = self.state["phase1_track_gains"]
            tracks = []
            for key, gains in self.state["learned_gains"].items():
                info = self.track_info[int(key)]
                tracks.append(
                    {
                        "catalog_id": info["catalog_id"],
                        "heard_in_music": int(key) in self.state["music_tracks"],
                        "gains": gains,
                        "curve": curve(_gains(gains)),
                        "phase1_gains": phase1[key],
                        "phase1_curve": curve(_gains(phase1[key])),
                    }
                )
            return {
                "code": self.code,
                "freqs": [round(float(f), 1) for f in CURVE_FREQS_HZ],
                "bands": list(DEFAULT_CENTER_FREQS_HZ),
                "tracks": tracks,
                "blind": blind,
                "n_comparisons": self.state.get("learned_n_comparisons", 0),
            }

    def close(self) -> None:
        """Release the database connection."""
        with self.lock:
            self.storage.close()


class Registry:
    """All participants, loaded on demand and evicted from memory when idle."""

    def __init__(self, settings: Settings) -> None:
        if not settings.template_path.exists():
            raise FileNotFoundError(
                f"{settings.template_path} not found; run `python -m web.prepare` first"
            )
        self.settings = settings
        self.clips = ClipCache(resolve=self._resolve)
        self._active: OrderedDict[str, Participant] = OrderedDict()
        self._lock = threading.Lock()
        self._reference: tuple[bytes, str] | None = None

    def _resolve(self, path: str) -> Path:
        """Find an audio file even if the data or music folder moved since prepare."""
        stored = Path(path)
        if stored.exists():
            return stored
        for folder in (self.settings.music_dir, self.settings.synthetic_dir):
            candidate = folder / stored.name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"audio file not found: {stored.name}")

    def create(self, mode: str, device: str) -> Participant:
        """Register a new participant."""
        with self._lock:
            code = new_code()
            while (self.settings.participants_dir / f"{code}.sqlite").exists():
                code = new_code()
            participant = Participant.create(self.settings, self.clips, code, mode, device)
            self._remember(participant)
            return participant

    def get(self, code: str) -> Participant:
        """Return the participant with ``code``, loading them from disk if needed."""
        code = code.strip().upper()
        if not CODE_PATTERN.match(code):
            raise UnknownParticipantError("unknown participant code")
        with self._lock:
            participant = self._active.get(code)
            if participant is None:
                path = self.settings.participants_dir / f"{code}.sqlite"
                if not path.exists():
                    raise UnknownParticipantError("unknown participant code")
                participant = Participant(self.settings, self.clips, path)
                self._remember(participant)
            self._active.move_to_end(code)
            participant.last_used = time.monotonic()
            return participant

    def delete(self, code: str) -> None:
        """Erase a participant's data for good (their right under LGPD/GDPR)."""
        participant = self.get(code)
        with self._lock, participant.lock:
            self._active.pop(participant.code, None)
            participant.storage.close()
            participant.path.unlink(missing_ok=True)

    def _remember(self, participant: Participant) -> None:
        self._active[participant.code] = participant
        now = time.monotonic()
        for code in list(self._active):
            if len(self._active) <= self.settings.max_active:
                break
            old = self._active[code]
            if now - old.last_used > IDLE_EVICT_S and old.lock.acquire(blocking=False):
                try:
                    old.storage.close()
                    del self._active[code]
                finally:
                    old.lock.release()

    def reference(self) -> tuple[bytes, str]:
        """Flat excerpt of a typical song, for setting the volume."""
        if self._reference is None:
            with Storage(self.settings.template_path) as storage:
                library = Library(storage)
                info = storage.get_meta(TRACKS_META_KEY, {})
                music = [t for t in library.tracks if info[str(t.id)]["kind"] == "music"]
                tracks = music or library.tracks
                contexts = np.array([library.context(t) for t in tracks])
                track = tracks[diverse_subset(contexts, 1)[0]]
            renderer = ClipRenderer(self.clips)
            audio = renderer.excerpt(track, np.zeros(len(DEFAULT_CENTER_FREQS_HZ)))
            self._reference = encode(audio, renderer.sample_rate, self.settings.study.audio.format)
        return self._reference
