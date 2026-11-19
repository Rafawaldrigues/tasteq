"""Study settings (``study.yaml``), the music catalog (``catalog.yaml``) and paths.

Paths come from the environment so the same code runs locally and in a
container:

- ``TASTEQ_DATA_DIR``: template database, synthetic sounds and one database
  per participant (default ``web/data``).
- ``TASTEQ_MUSIC_DIR``: folder with the catalog's audio files.
- ``TASTEQ_STUDY_CONFIG`` / ``TASTEQ_CATALOG``: override the two YAML files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

WEB_DIR = Path(__file__).resolve().parent

Stage = Literal["screening", "music", "blind"]


class ModeConfig(BaseModel):
    """Question counts of one study mode."""

    label: str
    minutes: int
    screening: int = Field(ge=1)
    music: int = Field(ge=1)
    blind: int = Field(ge=1)


class ScreeningConfig(BaseModel):
    """Phase 1 on synthetic sounds."""

    questions_per_sound: int = Field(default=2, ge=1)


class MusicConfig(BaseModel):
    """Phase 2 on the songs the participant chose."""

    questions_per_track: int = Field(default=4, ge=1)
    min_tracks: int = Field(default=3, ge=1)
    holdout_tracks: int = Field(default=2, ge=0)
    min_styles: int = Field(default=2, ge=1)


class AudioConfig(BaseModel):
    """Excerpts sent to the browser."""

    excerpt_seconds: float = Field(default=8.0, gt=1.0)
    format: Literal["flac", "wav", "mp3"] = "flac"
    synthetic_sounds: int = Field(default=8, ge=2)


class ContactConfig(BaseModel):
    """Where participants ask questions or request data removal."""

    url: str = ""
    email: str = ""


BLIND_CONDITIONS = ("learned", "flat", "phase1", "random")


class StudyConfig(BaseModel):
    """Everything in ``study.yaml``."""

    model_kind: str = "neural_adaptive"
    space: Literal["raw", "dct"] = "raw"
    modes: dict[str, ModeConfig]
    screening: ScreeningConfig = Field(default_factory=ScreeningConfig)
    music: MusicConfig = Field(default_factory=MusicConfig)
    blind_pairs: list[tuple[str, str]] = Field(
        default_factory=lambda: [("learned", "flat"), ("learned", "phase1"), ("learned", "random")]
    )
    audio: AudioConfig = Field(default_factory=AudioConfig)
    devices: dict[str, str]
    contact: ContactConfig = Field(default_factory=ContactConfig)

    @model_validator(mode="after")
    def _check_pairs(self) -> StudyConfig:
        for pair in self.blind_pairs:
            unknown = set(pair) - set(BLIND_CONDITIONS)
            if unknown or pair[0] == pair[1]:
                raise ValueError(f"invalid blind pair {pair}; use two of {BLIND_CONDITIONS}")
        return self


class CatalogTrack(BaseModel):
    """One song of the catalog, with everything its license asks us to show."""

    id: int
    file: str
    title: str
    artist: str
    style: str
    genre: str
    duration_s: float
    fma_url: str
    license: str
    license_url: str
    explicit: bool = False
    ai_generated: bool = False
    notes: str = ""
    internal_notes: str = ""  # for the study team, never sent to the browser


class Catalog(BaseModel):
    """Styles and songs from ``catalog.yaml``."""

    styles: dict[str, str]
    tracks: list[CatalogTrack]

    @model_validator(mode="after")
    def _check_styles(self) -> Catalog:
        for track in self.tracks:
            if track.style not in self.styles:
                raise ValueError(f"track {track.id} has unknown style {track.style!r}")
        return self

    def by_file(self) -> dict[str, CatalogTrack]:
        """Tracks keyed by file name."""
        return {t.file: t for t in self.tracks}


def load_study(path: Path) -> StudyConfig:
    """Read and validate ``study.yaml``."""
    return StudyConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_catalog(path: Path) -> Catalog:
    """Read and validate ``catalog.yaml``."""
    return Catalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Settings:
    """Configuration plus the folders it works in."""

    study: StudyConfig
    catalog: Catalog
    data_dir: Path
    music_dir: Path
    max_active: int = 64  # participants kept in memory; others are rebuilt on demand

    @property
    def template_path(self) -> Path:
        """Pre-analyzed database every participant starts from."""
        return self.data_dir / "template.sqlite"

    @property
    def synthetic_dir(self) -> Path:
        """Where the synthetic phase-1 sounds are written."""
        return self.data_dir / "synthetic"

    @property
    def participants_dir(self) -> Path:
        """One SQLite file per participant."""
        return self.data_dir / "participants"

    @classmethod
    def from_env(cls) -> Settings:
        """Read settings from the environment variables in the module docstring."""
        env = os.environ
        return cls(
            study=load_study(Path(env.get("TASTEQ_STUDY_CONFIG", WEB_DIR / "study.yaml"))),
            catalog=load_catalog(Path(env.get("TASTEQ_CATALOG", WEB_DIR / "catalog.yaml"))),
            data_dir=Path(env.get("TASTEQ_DATA_DIR", WEB_DIR / "data")),
            music_dir=Path(env.get("TASTEQ_MUSIC_DIR", WEB_DIR / "music")),
            max_active=int(env.get("TASTEQ_MAX_ACTIVE", "64")),
        )
