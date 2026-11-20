from pathlib import Path

from web.config import WEB_DIR, load_catalog
from web.deploy_space import stage


def test_stage_writes_a_space_with_header_dockerfile_and_songs(tmp_path: Path) -> None:
    tracks = load_catalog(WEB_DIR / "catalog.yaml").tracks
    music = tmp_path / "music-src"
    music.mkdir()
    (music / tracks[0].file).write_bytes(b"fake mp3")

    out = tmp_path / "space"
    missing = stage(out, music)

    assert missing == [t.file for t in tracks[1:]]
    assert (out / "music" / tracks[0].file).read_bytes() == b"fake mp3"
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\n")
    assert "sdk: docker" in readme
    assert "app_port: 7860" in readme
    assert "COPY music ./music" in (out / "Dockerfile").read_text(encoding="utf-8")
    assert (out / "src" / "tasteq" / "__init__.py").exists()
    assert (out / "web" / "backup.py").exists()
    assert not (out / "web" / "data").exists()  # local participants never ship
    assert not list(out.rglob("__pycache__"))
