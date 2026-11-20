r"""Assemble the web study as a Hugging Face Space and upload it.

The Space repository is separate from the code repository: this script copies
what the image needs (the library, ``web/``, the songs of the catalog) next to
the Space's own ``README.md`` header and ``Dockerfile`` (from ``web/space/``),
then uploads the folder. Run it again after any change to redeploy.

    hf auth login                       # once, with a write token
    python -m web.deploy_space --space you/tasteq-study \
        --music-dir fma_lineup --backup-repo you/tasteq-study-data

``--dry-run DIR`` only writes the folder, to inspect it. The backup token the
Space needs is read from ``TASTEQ_BACKUP_TOKEN`` and stored as the Space
secret ``HF_TOKEN``; without it the script says where to add it by hand.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from web.config import WEB_DIR, load_catalog

ROOT = WEB_DIR.parent
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "data", ".pytest_cache", "space")


def stage(target: Path, music_dir: Path) -> list[str]:
    """Write the Space's files into ``target``; return catalog files missing from ``music_dir``."""
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "src", target / "src", ignore=SKIP, dirs_exist_ok=True)
    shutil.copytree(WEB_DIR, target / "web", ignore=SKIP, dirs_exist_ok=True)
    for name in ("pyproject.toml", "LICENSE"):
        shutil.copyfile(ROOT / name, target / name)
    shutil.copyfile(WEB_DIR / "space" / "README.md", target / "README.md")
    shutil.copyfile(WEB_DIR / "space" / "Dockerfile", target / "Dockerfile")

    music = target / "music"
    music.mkdir(exist_ok=True)
    missing = []
    for track in load_catalog(WEB_DIR / "catalog.yaml").tracks:
        source = music_dir / track.file
        if source.exists():
            shutil.copyfile(source, music / track.file)
        else:
            missing.append(track.file)
    return missing


def deploy(folder: Path, space: str, backup_repo: str | None) -> None:
    """Create the Space if needed, upload ``folder`` and configure the backup."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        repo_id=space,
        repo_type="space",
        folder_path=str(folder),
        commit_message="Deploy the listening study",
        delete_patterns=["src/**", "web/**", "music/**"],
    )
    if backup_repo:
        api.create_repo(backup_repo, repo_type="dataset", private=True, exist_ok=True)
        api.add_space_variable(space, "TASTEQ_BACKUP_REPO", backup_repo)
        token = os.environ.get("TASTEQ_BACKUP_TOKEN")
        if token:
            api.add_space_secret(space, "HF_TOKEN", token)
        else:
            print(
                "Add the secret HF_TOKEN (a token that can write to "
                f"{backup_repo}) in the Space settings, then restart the Space."
            )
    print(f"Deployed: https://huggingface.co/spaces/{space}")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--space", help="Space id, e.g. you/tasteq-study")
    parser.add_argument("--music-dir", type=Path, default=ROOT / "fma_lineup")
    parser.add_argument("--backup-repo", help="private dataset for the answers")
    parser.add_argument("--dry-run", type=Path, metavar="DIR", help="only write the folder")
    args = parser.parse_args(argv)

    if args.dry_run:
        missing = stage(args.dry_run, args.music_dir)
        print(f"wrote {args.dry_run}")
    else:
        if not args.space:
            parser.error("--space is required unless --dry-run is given")
        with tempfile.TemporaryDirectory() as tmp:
            missing = stage(Path(tmp), args.music_dir)
            if missing:
                print(f"missing songs in {args.music_dir}: {missing}", file=sys.stderr)
                return 1
            deploy(Path(tmp), args.space, args.backup_repo)
    if missing:
        print(f"missing songs in {args.music_dir}: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
