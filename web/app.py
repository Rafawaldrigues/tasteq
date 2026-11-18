"""HTTP API of the web study, plus the static front end.

Run locally with::

    TASTEQ_MUSIC_DIR=/path/to/music python -m web.prepare
    TASTEQ_MUSIC_DIR=/path/to/music uvicorn web.app:app

Handlers are plain (sync) functions, so FastAPI runs them in its thread pool;
model fitting and rendering never block the event loop, and each
participant's lock keeps their own requests in order.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from tasteq.storage import Storage
from web.config import WEB_DIR, Settings
from web.prepare import TRACKS_META_KEY
from web.study import Registry, StudyError

Rating = Annotated[int, Field(ge=1, le=5)]
NO_STORE = {"Cache-Control": "no-store"}


class NewParticipant(BaseModel):
    """Consent screen."""

    mode: str
    device: str
    consent: bool
    adult: bool


class AnswerIn(BaseModel):
    """An A/B answer and how the participant listened before giving it."""

    qid: str
    answer: Literal["first", "second", "tie"]
    plays_first: int | None = Field(default=None, ge=0)
    plays_second: int | None = Field(default=None, ge=0)
    switches: int | None = Field(default=None, ge=0)
    listen_ms: int | None = Field(default=None, ge=0)


class StylesIn(BaseModel):
    """Music styles the participant likes."""

    styles: list[str]


class QuestionnaireIn(BaseModel):
    """Short questionnaire at the end."""

    difficulty: Rating
    fatigue: Rating
    experience: Literal["none", "listener", "amateur", "professional"]
    uses_eq: Literal["never", "sometimes", "often"]
    comment: str = Field(default="", max_length=2000)


def _limit_threads() -> None:
    """Keep each model fit to a few threads so concurrent participants share the CPU."""
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(int(os.environ.get("TASTEQ_TORCH_THREADS", "2")))


def public_config(settings: Settings) -> dict[str, Any]:
    """Return what the front end needs for the consent, mode and style screens."""
    with Storage(settings.template_path) as storage:
        info = storage.get_meta(TRACKS_META_KEY, {})
    available = {v["catalog_id"] for v in info.values() if v["kind"] == "music"}
    catalog = settings.catalog
    per_style = Counter(t.style for t in catalog.tracks if t.id in available)
    study = settings.study
    return {
        "modes": {
            key: {
                "label": m.label,
                "minutes": m.minutes,
                "counts": {"screening": m.screening, "music": m.music, "blind": m.blind},
            }
            for key, m in study.modes.items()
        },
        "devices": study.devices,
        "styles": [
            {"id": key, "label": label, "n_tracks": per_style.get(key, 0)}
            for key, label in catalog.styles.items()
        ],
        "min_styles": study.music.min_styles,
        "tracks": [
            {**t.model_dump(exclude={"file", "internal_notes"}), "available": t.id in available}
            for t in catalog.tracks
        ],
        "contact": study.contact.model_dump(),
        "model_kind": study.model_kind,
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app; ``settings`` default to the environment."""
    settings = settings or Settings.from_env()
    registry = Registry(settings)
    config = public_config(settings)
    registry.reference()  # render once at startup, not on the first visitor's click
    _limit_threads()

    app = FastAPI(title="tasteq study", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.registry = registry

    @app.exception_handler(StudyError)
    def study_error(_: Request, error: StudyError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=error.status_code)

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return config

    @app.get("/api/reference")
    def reference() -> Response:
        data, media_type = registry.reference()
        return Response(data, media_type=media_type)

    @app.post("/api/participants")
    def create(body: NewParticipant) -> dict[str, Any]:
        if not (body.consent and body.adult):
            raise HTTPException(400, "consent and age confirmation are required")
        return registry.create(body.mode, body.device).status()

    @app.get("/api/p/{code}")
    def status(code: str) -> dict[str, Any]:
        return registry.get(code).status()

    @app.post("/api/p/{code}/volume")
    def volume(code: str) -> dict[str, Any]:
        return registry.get(code).volume_set()

    @app.post("/api/p/{code}/question")
    def question(code: str) -> dict[str, Any]:
        return registry.get(code).question()

    @app.get("/api/p/{code}/audio/{qid}/{slot}")
    def audio(code: str, qid: str, slot: int) -> Response:
        data, media_type = registry.get(code).audio(qid, slot)
        return Response(data, media_type=media_type, headers=NO_STORE)

    @app.post("/api/p/{code}/answer")
    def answer(code: str, body: AnswerIn) -> dict[str, Any]:
        listening = body.model_dump(exclude={"qid", "answer"}, exclude_none=True)
        return registry.get(code).answer(body.qid, body.answer, listening)

    @app.post("/api/p/{code}/styles")
    def styles(code: str, body: StylesIn) -> dict[str, Any]:
        return registry.get(code).choose_styles(body.styles)

    @app.post("/api/p/{code}/questionnaire")
    def questionnaire(code: str, body: QuestionnaireIn) -> dict[str, Any]:
        return registry.get(code).submit_questionnaire(body.model_dump())

    @app.get("/api/p/{code}/results")
    def results(code: str) -> dict[str, Any]:
        return registry.get(code).results()

    @app.post("/api/p/{code}/withdraw")
    def withdraw(code: str) -> dict[str, Any]:
        participant = registry.get(code)
        participant.withdraw()
        return participant.status()

    @app.delete("/api/p/{code}")
    def delete(code: str) -> dict[str, str]:
        registry.delete(code)
        return {"deleted": code.upper()}

    app.mount("/", StaticFiles(directory=WEB_DIR / "static", html=True), name="static")
    return app
