import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from web.backup import REMOTE_DIR, Backup


class FakeHub:
    """Just enough of ``HfApi`` for the backup: a dataset kept in a local folder."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.created: list[tuple[str, bool]] = []
        self.uploads = 0
        self.squashes = 0
        self.fail_listing = False

    def create_repo(self, repo_id: str, repo_type: str, private: bool, exist_ok: bool) -> None:
        assert repo_type == "dataset"
        self.created.append((repo_id, private))
        self.root.mkdir(parents=True, exist_ok=True)

    def list_repo_files(self, repo_id: str, repo_type: str) -> list[str]:
        if self.fail_listing:
            raise ConnectionError("hub unreachable")
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file())

    def hf_hub_download(self, repo_id: str, filename: str, repo_type: str, local_dir: str) -> str:
        target = Path(local_dir) / "fake-hub" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / filename, target)
        return str(target)

    def upload_folder(self, **kwargs: object) -> None:
        self.uploads += 1
        remote = self.root / str(kwargs["path_in_repo"])
        remote.mkdir(parents=True, exist_ok=True)
        local = Path(str(kwargs["folder_path"]))
        for old in remote.glob("*.sqlite"):  # delete_patterns=["*.sqlite"]
            if not (local / old.name).exists():
                old.unlink()
        for new in local.glob("*.sqlite"):
            shutil.copyfile(new, remote / new.name)

    def super_squash_history(self, repo_id: str, repo_type: str) -> None:
        self.squashes += 1


def participant(folder: Path, code: str, answers: int) -> Path:
    path = folder / f"{code}.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("create table if not exists answers (n integer)")
        db.executemany("insert into answers values (?)", [(i,) for i in range(answers)])
    return path


def answers(path: Path) -> int:
    with sqlite3.connect(path) as db:
        return int(db.execute("select count(*) from answers").fetchone()[0])


def test_round_trip_through_a_restart(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    first = Backup(tmp_path / "run1", "me/data", api=hub)
    assert first.restore() == 0
    assert hub.created == [("me/data", True)]  # private
    participant(first.folder, "AAAA-1111", 3)
    participant(first.folder, "BBBB-2222", 5)
    assert first.sync()
    assert not first.sync()  # nothing changed, nothing sent
    assert hub.uploads == 1

    # The Space restarts with an empty disk.
    second = Backup(tmp_path / "run2", "me/data", api=hub)
    assert second.restore() == 2
    assert answers(second.folder / "AAAA-1111.sqlite") == 3
    assert answers(second.folder / "BBBB-2222.sqlite") == 5


def test_deleted_participant_leaves_the_dataset_and_its_history(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    backup = Backup(tmp_path / "run", "me/data", api=hub)
    backup.restore()
    participant(backup.folder, "AAAA-1111", 1)
    keep = participant(backup.folder, "BBBB-2222", 1)
    backup.sync()
    (backup.folder / "AAAA-1111.sqlite").unlink()
    assert backup.sync()
    assert sorted(p.name for p in (hub.root / REMOTE_DIR).iterdir()) == [keep.name]
    assert hub.squashes == 1


def test_changed_answers_are_pushed_again(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    backup = Backup(tmp_path / "run", "me/data", api=hub)
    backup.restore()
    path = participant(backup.folder, "AAAA-1111", 1)
    backup.sync()
    participant(backup.folder, "AAAA-1111", 2)
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
    assert backup.sync()
    assert answers(hub.root / REMOTE_DIR / "AAAA-1111.sqlite") == 3


def test_no_sync_before_a_successful_restore(tmp_path: Path) -> None:
    hub = FakeHub(tmp_path / "hub")
    seed = Backup(tmp_path / "seed", "me/data", api=hub)
    seed.restore()
    participant(seed.folder, "AAAA-1111", 4)
    seed.sync()

    hub.fail_listing = True
    backup = Backup(tmp_path / "run", "me/data", api=hub)
    with pytest.raises(ConnectionError):
        backup.restore()
    participant(backup.folder, "CCCC-3333", 1)
    # Syncing now would wipe AAAA from the dataset; it must not happen.
    assert not backup.sync()
    assert (hub.root / REMOTE_DIR / "AAAA-1111.sqlite").exists()


def test_backup_is_off_without_a_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TASTEQ_BACKUP_REPO", raising=False)
    assert Backup.from_env(tmp_path) is None
