"""The web study backend: plans, isolation between participants, blind test and export."""

from __future__ import annotations

import collections
import csv
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("librosa")

from fastapi.testclient import TestClient

from tasteq.synth import write_demo_library
from web.config import (
    WEB_DIR,
    Catalog,
    Settings,
    StudyConfig,
    load_catalog,
    load_study,
)

STYLES = {"a": "Style A", "b": "Style B", "c": "Style C"}


def study_config(**overrides: Any) -> StudyConfig:
    data: dict[str, Any] = {
        "model_kind": "random",  # fast; the GP is exercised by tests/test_models.py
        "modes": {
            "quick": {"label": "Quick", "minutes": 1, "screening": 4, "music": 6, "blind": 6},
            "medium": {"label": "Medium", "minutes": 2, "screening": 6, "music": 10, "blind": 8},
        },
        "screening": {"questions_per_sound": 2},
        "music": {"questions_per_track": 2, "min_tracks": 3, "holdout_tracks": 2, "min_styles": 2},
        "audio": {"excerpt_seconds": 3.0, "format": "flac", "synthetic_sounds": 4},
        "devices": {"closed": "Closed", "earbuds": "Earbuds"},
    }
    data.update(overrides)
    return StudyConfig.model_validate(data)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """A template with 4 synthetic sounds and 9 short "songs" in 3 styles."""
    from web.prepare import prepare

    root = tmp_path_factory.mktemp("study")
    music = root / "music"
    tracks = []
    for i, path in enumerate(write_demo_library(music, n_tracks=9, duration_s=6.0, seed=1)):
        name = f"song_{i:02d}.wav"
        path.rename(music / name)
        tracks.append(
            {
                "id": i + 1,
                "file": name,
                "title": f"Song {i + 1}",
                "artist": "Artist",
                "style": "abc"[i % 3],
                "genre": "Test",
                "duration_s": 6,
                "fma_url": "https://example.org",
                "license": "CC0 1.0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            }
        )
    settings = Settings(
        study=study_config(),
        catalog=Catalog.model_validate({"styles": STYLES, "tracks": tracks}),
        data_dir=root / "data",
        music_dir=music,
    )
    prepare(settings)
    return settings


@pytest.fixture
def settings(prepared: Settings, tmp_path: Path) -> Settings:
    """Each test gets its own participants folder next to the shared template."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "template.sqlite").symlink_to(prepared.template_path)
    (data / "synthetic").symlink_to(prepared.synthetic_dir)
    return Settings(prepared.study, prepared.catalog, data, prepared.music_dir)


@pytest.fixture
def client(settings: Settings) -> TestClient:
    from web.app import create_app

    return TestClient(create_app(settings))


def start(client: TestClient, mode: str = "quick") -> str:
    response = client.post(
        "/api/participants",
        json={"mode": mode, "device": "closed", "consent": True, "adult": True},
    )
    response.raise_for_status()
    code: str = response.json()["code"]
    client.post(f"/api/p/{code}/volume").raise_for_status()
    return code


def answer_stage(client: TestClient, code: str, answers: str = "fst") -> dict[str, Any]:
    """Answer questions until the participant leaves the current question stage."""
    choices = {"f": "first", "s": "second", "t": "tie"}
    stage = client.get(f"/api/p/{code}").json()["stage"]
    assert stage in ("screening", "music", "blind")
    i = 0
    while True:
        status: dict[str, Any] = client.get(f"/api/p/{code}").json()
        if status["stage"] != stage:
            return status
        question = client.post(f"/api/p/{code}/question").json()
        for url in question["audio"]:
            audio = client.get(url)
            assert audio.status_code == 200
            assert audio.headers["content-type"] == "audio/flac"
        client.post(
            f"/api/p/{code}/answer",
            json={"qid": question["qid"], "answer": choices[answers[i % len(answers)]]},
        ).raise_for_status()
        i += 1


def run_to_end(client: TestClient, code: str, styles: tuple[str, ...] = ("a", "b")) -> None:
    assert answer_stage(client, code)["stage"] == "styles"
    client.post(f"/api/p/{code}/styles", json={"styles": list(styles)}).raise_for_status()
    assert answer_stage(client, code)["stage"] == "blind"
    assert answer_stage(client, code)["stage"] == "questionnaire"
    client.post(
        f"/api/p/{code}/questionnaire",
        json={"difficulty": 3, "fatigue": 2, "experience": "listener", "uses_eq": "never"},
    ).raise_for_status()


def db_rows(settings: Settings, code: str, sql: str) -> list[tuple[Any, ...]]:
    import sqlite3

    with sqlite3.connect(settings.participants_dir / f"{code}.sqlite") as conn:
        return conn.execute(sql).fetchall()


# --- configuration files ------------------------------------------------------


def test_shipped_config_and_catalog_are_valid() -> None:
    study = load_study(WEB_DIR / "study.yaml")
    catalog = load_catalog(WEB_DIR / "catalog.yaml")
    assert study.model_kind == "gp"
    assert {(m.screening, m.music, m.blind) for m in study.modes.values()} == {
        (8, 12, 10),
        (12, 24, 16),
        (16, 40, 24),
    }
    assert len(catalog.tracks) == 28
    licenses = collections.Counter(t.license for t in catalog.tracks)
    assert licenses == {"CC BY 4.0": 20, "CC0 1.0": 8}
    per_style = collections.Counter(t.style for t in catalog.tracks)
    # Any two styles leave enough songs for phase 2 plus the held-out ones.
    smallest_two = sum(sorted(per_style.values())[:2])
    assert smallest_two >= study.music.min_tracks + study.music.holdout_tracks
    assert [t.id for t in catalog.tracks if t.ai_generated] == [18]


def test_invalid_blind_pair_is_rejected() -> None:
    with pytest.raises(ValueError, match="blind pair"):
        study_config(blind_pairs=[["learned", "learned"]])


# --- the full flow ------------------------------------------------------------


@pytest.mark.parametrize("mode", ["quick", "medium"])
def test_phase_counts_follow_the_mode(client: TestClient, settings: Settings, mode: str) -> None:
    code = start(client, mode)
    run_to_end(client, code)
    counts = settings.study.modes[mode]

    per_kind = dict(
        db_rows(
            settings,
            code,
            "SELECT s.kind, count(*) FROM comparisons c JOIN sessions s ON s.id = c.session_id "
            "GROUP BY s.kind",
        )
    )
    assert per_kind == {"screening": counts.screening, "music": counts.music}
    assert db_rows(settings, code, "SELECT count(*) FROM blind_trials")[0][0] == counts.blind
    logged = dict(db_rows(settings, code, "SELECT stage, count(*) FROM web_answers GROUP BY stage"))
    assert logged == {"screening": counts.screening, "music": counts.music, "blind": counts.blind}
    assert client.get(f"/api/p/{code}").json()["stage"] == "done"


def test_music_phase_uses_chosen_styles_and_holds_songs_out(
    client: TestClient, settings: Settings
) -> None:
    code = start(client)
    answer_stage(client, code)
    client.post(f"/api/p/{code}/styles", json={"styles": ["a", "c"]}).raise_for_status()
    answer_stage(client, code)
    answer_stage(client, code)

    state = json.loads(
        db_rows(settings, code, "SELECT value FROM meta WHERE key = 'web_state'")[0][0]
    )
    info = json.loads(
        db_rows(settings, code, "SELECT value FROM meta WHERE key = 'web_tracks'")[0][0]
    )
    style = {t.id: t.style for t in settings.catalog.tracks}
    chosen = state["music_tracks"] + state["holdout_tracks"]
    assert {style[info[str(i)]["catalog_id"]] for i in chosen} <= {"a", "c"}
    assert len(state["music_tracks"]) >= settings.study.music.min_tracks
    assert len(state["holdout_tracks"]) == 2
    assert not set(state["music_tracks"]) & set(state["holdout_tracks"])

    music_tracks = {
        r[0]
        for r in db_rows(
            settings,
            code,
            "SELECT DISTINCT c.track_id FROM comparisons c JOIN sessions s ON s.id = c.session_id "
            "WHERE s.kind = 'music'",
        )
    }
    assert music_tracks == set(state["music_tracks"])
    blind_tracks = {r[0] for r in db_rows(settings, code, "SELECT track_id FROM blind_trials")}
    assert set(state["holdout_tracks"]) <= blind_tracks  # generalization is tested


def test_too_few_styles_is_rejected(client: TestClient) -> None:
    code = start(client)
    answer_stage(client, code)
    response = client.post(f"/api/p/{code}/styles", json={"styles": ["a"]})
    assert response.status_code == 400
    assert client.get(f"/api/p/{code}").json()["stage"] == "styles"


# --- blind test ---------------------------------------------------------------


def test_blind_test_is_frozen_and_never_trains_the_model(
    client: TestClient, settings: Settings
) -> None:
    code = start(client)
    answer_stage(client, code)
    client.post(f"/api/p/{code}/styles", json={"styles": ["a", "b"]}).raise_for_status()
    answer_stage(client, code)
    n_before = db_rows(settings, code, "SELECT count(*) FROM comparisons")[0][0]
    assert answer_stage(client, code)["stage"] == "questionnaire"
    assert db_rows(settings, code, "SELECT count(*) FROM comparisons")[0][0] == n_before

    state = json.loads(
        db_rows(settings, code, "SELECT value FROM meta WHERE key = 'web_state'")[0][0]
    )
    trials = db_rows(
        settings,
        code,
        "SELECT track_id, condition_a, condition_b, gains_a, gains_b, n_comparisons, session_id "
        "FROM blind_trials",
    )
    kinds = dict(db_rows(settings, code, "SELECT id, kind FROM sessions"))
    pairs = {frozenset(p) for p in settings.study.blind_pairs}
    seen = set()
    for track_id, cond_a, cond_b, gains_a, gains_b, n_comparisons, session_id in trials:
        assert kinds[session_id] == "blindtest"
        assert frozenset((cond_a, cond_b)) in pairs
        seen.add(frozenset((cond_a, cond_b)))
        assert n_comparisons == n_before
        gains = {cond_a: json.loads(gains_a), cond_b: json.loads(gains_b)}
        np.testing.assert_allclose(gains["learned"], state["learned_gains"][str(track_id)])
        if "flat" in gains:
            assert not any(gains["flat"])
        if "phase1" in gains:
            np.testing.assert_allclose(gains["phase1"], state["phase1_gains"])
        if "random" in gains:
            # Same RMS as the learned EQ, minus whatever the ±10 dB clip removes.
            rms = lambda g: float(np.sqrt(np.mean(np.square(g))))  # noqa: E731
            assert 0 < rms(gains["random"]) <= rms(gains["learned"]) + 1e-3
    assert seen == pairs

    details = [
        json.loads(d)
        for (d,) in db_rows(settings, code, "SELECT detail FROM web_answers WHERE stage = 'blind'")
    ]
    assert all("heard_in_music" in d for d in details)

    client.post(
        f"/api/p/{code}/questionnaire",
        json={"difficulty": 1, "fatigue": 1, "experience": "none", "uses_eq": "often"},
    ).raise_for_status()
    results = client.get(f"/api/p/{code}/results").json()
    assert [r["pair"] for r in results["blind"]] == [list(p) for p in settings.study.blind_pairs]
    assert sum(r["wins"] + r["losses"] + r["ties"] for r in results["blind"]) == len(trials)
    assert len(results["phase1"]["curve"]) == len(results["freqs"])


# --- isolation and robustness ---------------------------------------------------


def test_participants_are_isolated_under_concurrency(settings: Settings) -> None:
    from web.study import Registry

    registry = Registry(settings)
    participants = [registry.create("quick", "closed") for _ in range(3)]
    errors: list[BaseException] = []

    def run(index: int) -> None:
        try:
            p = participants[index]
            p.volume_set()
            answers = ["first", "second", "tie"]
            while p.status()["stage"] == "screening":
                q = p.question()
                p.answer(q["qid"], answers[index])  # type: ignore[arg-type]
        except BaseException as error:  # surfaced below
            errors.append(error)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors

    for index, p in enumerate(participants):
        outcomes = {r[0] for r in db_rows(settings, p.code, "SELECT outcome FROM comparisons")}
        rows = db_rows(settings, p.code, "SELECT count(*) FROM comparisons")[0][0]
        assert rows == settings.study.modes["quick"].screening
        assert p.tuner.n_comparisons == rows
        # Each participant only ever answered one way; nobody else's answers leaked in.
        if index == 2:
            assert outcomes == {"tie"}
        else:
            assert "tie" not in outcomes


def test_stale_answers_and_unknown_codes(client: TestClient) -> None:
    code = start(client)
    question = client.post(f"/api/p/{code}/question").json()
    # Asking again (a reload) returns the same open question.
    assert client.post(f"/api/p/{code}/question").json()["qid"] == question["qid"]
    ok = client.post(f"/api/p/{code}/answer", json={"qid": question["qid"], "answer": "first"})
    assert ok.status_code == 200
    again = client.post(f"/api/p/{code}/answer", json={"qid": question["qid"], "answer": "first"})
    assert again.status_code == 409
    assert client.get(f"/api/p/{code}/audio/{question['qid']}/1").status_code == 409
    assert client.get("/api/p/ZZZZ-ZZZZ").status_code == 404
    assert client.get("/api/p/..%2F..%2Fetc").status_code == 404
    assert (
        client.post(
            "/api/participants",
            json={"mode": "quick", "device": "closed", "consent": False, "adult": True},
        ).status_code
        == 400
    )


def test_progress_survives_a_server_restart(settings: Settings) -> None:
    from web.study import Registry

    first = Registry(settings).create("quick", "closed")
    first.volume_set()
    for _ in range(3):
        first.answer(first.question()["qid"], "first")
    first.close()

    again = Registry(settings).get(first.code)
    assert again.status()["position"] == 3
    assert again.tuner.n_comparisons == 3
    again.answer(again.question()["qid"], "second")
    assert again.status()["stage"] == "styles"


def test_withdraw_keeps_data_and_delete_erases_it(client: TestClient, settings: Settings) -> None:
    kept = start(client)
    client.post(
        f"/api/p/{kept}/answer",
        json={"qid": client.post(f"/api/p/{kept}/question").json()["qid"], "answer": "first"},
    ).raise_for_status()
    assert client.post(f"/api/p/{kept}/withdraw").json()["stage"] == "withdrawn"
    assert (settings.participants_dir / f"{kept}.sqlite").exists()

    gone = start(client)
    assert client.delete(f"/api/p/{gone}").status_code == 200
    assert not (settings.participants_dir / f"{gone}.sqlite").exists()
    assert client.get(f"/api/p/{gone}").status_code == 404


# --- export --------------------------------------------------------------------


def test_export_includes_complete_and_partial_participants(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    from web.export import export

    done = start(client)
    run_to_end(client, done)
    partial = start(client)
    for _ in range(2):
        q = client.post(f"/api/p/{partial}/question").json()
        client.post(
            f"/api/p/{partial}/answer",
            json={
                "qid": q["qid"],
                "answer": "tie",
                "plays_first": 2,
                "plays_second": 1,
                "switches": 3,
                "listen_ms": 4200,
            },
        ).raise_for_status()
    left = start(client)
    client.post(f"/api/p/{left}/withdraw").raise_for_status()

    counts = export(settings.participants_dir, tmp_path / "out")
    quick = settings.study.modes["quick"]
    assert counts == {
        "participants": 3,
        "comparisons": quick.screening + quick.music + 2,
        "blind_trials": quick.blind,
        "answers": quick.screening + quick.music + quick.blind + 2,
    }

    def read(name: str) -> list[dict[str, str]]:
        with (tmp_path / "out" / f"{name}.csv").open() as f:
            return list(csv.DictReader(f))

    status = {r["code"]: r["status"] for r in read("participants")}
    assert status == {done: "completed", partial: "in_progress", left: "withdrawn"}
    finished = next(r for r in read("participants") if r["code"] == done)
    assert finished["q_difficulty"] == "3"
    assert finished["answered_blind"] == str(quick.blind)

    comparisons = [r for r in read("comparisons") if r["code"] == partial]
    assert {r["session_kind"] for r in comparisons} == {"screening"}
    assert comparisons[0]["listen_ms"] == "4200"
    assert comparisons[0]["switches"] == "3"
    blind = read("blind_trials")
    assert {r["heard_in_music"] for r in blind} == {"True", "False"}
    assert all(r["catalog_id"] for r in blind)
