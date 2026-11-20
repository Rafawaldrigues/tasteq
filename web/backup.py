"""Keep the participants' databases in a private Hugging Face dataset.

Free hosting (a Hugging Face Space on the free CPU tier) has no persistent
disk: every restart, sleep or rebuild starts from the image again. So the
server restores ``participants/*.sqlite`` from a private dataset when it
starts, and pushes changes back every few minutes and when it shuts down.

A participant who deletes their answers is removed from the dataset on the
next sync, and the dataset's history is squashed so no earlier commit keeps
their file.

Configuration, all from the environment (Space secrets and variables):

- ``TASTEQ_BACKUP_REPO``: dataset id, e.g. ``someone/tasteq-study-data``.
  Without it, backup is off (local runs).
- ``HF_TOKEN``: a token with write access to that dataset (a Space secret).
- ``TASTEQ_BACKUP_EVERY``: seconds between syncs, default 300.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REMOTE_DIR = "participants"


def _snapshot(source: Path, target: Path) -> None:
    """Copy a live SQLite database consistently (the online backup API)."""
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


class Backup:
    """Restore and sync one folder of participant databases with a dataset repo."""

    def __init__(
        self,
        folder: Path,
        repo_id: str,
        token: str | None = None,
        every_s: float = 300.0,
        api: Any = None,
    ) -> None:
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
        self.folder = folder
        self.repo_id = repo_id
        self.every_s = every_s
        self.api = api
        self.restored = False
        self._synced: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_env(cls, folder: Path) -> Backup | None:
        """Return a backup configured from the environment, or ``None`` when it is off."""
        repo = os.environ.get("TASTEQ_BACKUP_REPO", "").strip()
        if not repo:
            return None
        every = float(os.environ.get("TASTEQ_BACKUP_EVERY", "300"))
        return cls(folder, repo, token=os.environ.get("HF_TOKEN") or None, every_s=every)

    # ---------------------------------------------------------------- restore

    def restore(self) -> int:
        """Create the private dataset if needed and download every participant missing here.

        Sync stays disabled unless this succeeds: syncing an empty folder
        after a failed restore would delete everyone's answers from the dataset.
        """
        self.folder.mkdir(parents=True, exist_ok=True)
        self.api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)
        remote = [
            f
            for f in self.api.list_repo_files(self.repo_id, repo_type="dataset")
            if f.startswith(f"{REMOTE_DIR}/") and f.endswith(".sqlite")
        ]
        restored = 0
        for name in remote:
            local = self.folder / Path(name).name
            if local.exists():
                continue
            downloaded = self.api.hf_hub_download(
                self.repo_id, name, repo_type="dataset", local_dir=tempfile.gettempdir()
            )
            shutil.copyfile(downloaded, local)
            restored += 1
        with self._lock:
            self._synced = self._fingerprint()
        self.restored = True
        log.info("restored %d of %d participants from %s", restored, len(remote), self.repo_id)
        return restored

    # ---------------------------------------------------------------- sync

    def _fingerprint(self) -> dict[str, tuple[int, int]]:
        return {
            p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in self.folder.glob("*.sqlite")
        }

    def sync(self) -> bool:
        """Push changed databases and drop deleted ones; return whether anything was sent."""
        if not self.restored:
            return False
        with self._lock:
            current = self._fingerprint()
            if current == self._synced:
                return False
            removed = set(self._synced) - set(current)
            with tempfile.TemporaryDirectory() as tmp:
                staging = Path(tmp)
                for name in current:
                    _snapshot(self.folder / name, staging / name)
                self.api.upload_folder(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    folder_path=str(staging),
                    path_in_repo=REMOTE_DIR,
                    delete_patterns=["*.sqlite"],
                    commit_message=f"sync {len(current)} participants",
                )
            if removed:
                # Deleted answers must not survive in an older commit.
                self.api.super_squash_history(self.repo_id, repo_type="dataset")
            self._synced = current
        log.info("synced %d participants (%d removed)", len(current), len(removed))
        return True

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Sync in a background thread every ``every_s`` seconds."""

        def loop() -> None:
            while not self._stop.wait(self.every_s):
                try:
                    if not self.restored:
                        self.restore()  # startup restore failed; keep trying
                    self.sync()
                except Exception:  # keep serving; the next round retries
                    log.exception("backup sync failed")

        self._thread = threading.Thread(target=loop, name="tasteq-backup", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the loop and push whatever changed since the last sync."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        try:
            self.sync()
        except Exception:
            log.exception("final backup sync failed")
