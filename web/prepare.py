"""Build the template database every participant's database is copied from.

The synthetic phase-1 sounds and all catalog songs are analyzed together, so
the context projection (and therefore what "a similar song" means to the
model) is the same for every participant. Run once per deployment::

    python -m web.prepare            # skips the work if the template exists
    python -m web.prepare --force    # rebuild, e.g. after changing the music
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from tasteq.analysis import ContextProjector, analyze_track
from tasteq.library import CONTEXT_META_KEY
from tasteq.storage import Storage
from web.config import Settings
from web.loops import write_screening_loops

log = logging.getLogger(__name__)

TRACKS_META_KEY = "web_tracks"
TEMPLATE_META_KEY = "web_template"
SYNTHETIC_SECONDS = 30.0


def synthetic_files(settings: Settings) -> list[Path]:
    """Return the synthetic sounds, writing them on first use."""
    folder = settings.synthetic_dir
    n = settings.study.audio.synthetic_sounds
    # Older templates used noisy stress signals (synthetic_*.wav); replace them.
    for old in folder.glob("synthetic_*.wav"):
        old.unlink()
    existing = sorted(folder.glob("loop_*.wav"))
    if len(existing) != n:
        for path in existing:
            path.unlink()
        return write_screening_loops(folder, n_loops=n, duration_s=SYNTHETIC_SECONDS, seed=0)
    return existing


def prepare(settings: Settings, force: bool = False) -> Path:
    """Write ``settings.template_path`` and return it."""
    template = settings.template_path
    if template.exists() and not force:
        log.info("template exists: %s", template)
        return template

    excerpt_s = settings.study.audio.excerpt_seconds
    sources: list[tuple[Path, dict[str, object]]] = [
        (path, {"kind": "synthetic", "catalog_id": None, "name": f"Loop {i + 1}"})
        for i, path in enumerate(synthetic_files(settings))
    ]
    missing = []
    for track in settings.catalog.tracks:
        path = settings.music_dir / track.file
        if path.exists():
            sources.append((path, {"kind": "music", "catalog_id": track.id, "name": track.title}))
        else:
            missing.append(track.file)
    if missing:
        log.warning(
            "%d catalog files not found in %s: %s", len(missing), settings.music_dir, missing
        )

    partial = template.with_suffix(".partial")
    partial.unlink(missing_ok=True)
    with Storage(partial) as storage:
        tracks: dict[str, dict[str, object]] = {}
        for path, info in sources:
            log.info("analyzing %s", path.name)
            track_id = storage.upsert_track(analyze_track(path.resolve(), excerpt_s))
            tracks[str(track_id)] = {**info, "file": path.name}
        features = np.array([t.analysis.feature_vector() for t in storage.tracks()])
        storage.set_meta(CONTEXT_META_KEY, ContextProjector.fit(features).to_dict())
        storage.set_meta(TRACKS_META_KEY, tracks)
        storage.set_meta(
            TEMPLATE_META_KEY,
            {
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "excerpt_seconds": excerpt_s,
                "missing": missing,
            },
        )
    os.replace(partial, template)
    log.info("wrote %s (%d tracks, %d missing)", template, len(sources), len(missing))
    return template


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="rebuild even if it exists")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    prepare(Settings.from_env(), force=args.force)


if __name__ == "__main__":
    main()
