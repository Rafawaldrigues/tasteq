import sys
from pathlib import Path

import numpy as np
import pytest

from tasteq import playback
from tasteq.playback import CommandPlayer, PlaybackUnavailableError, default_player


def test_command_player_runs_and_stops(tmp_path: Path) -> None:
    # A "player" that just sleeps stands in for pw-play/aplay.
    player = CommandPlayer((sys.executable, "-c", "import time, sys; time.sleep(5)"))
    player.start(np.zeros((4410, 2)), 44_100)
    assert player._process is not None
    player.stop()
    assert player._process is None


def test_command_player_waits_for_completion() -> None:
    player = CommandPlayer((sys.executable, "-c", "pass"))
    player.start(np.zeros(4410), 44_100)
    player.wait()
    player.stop()


def test_no_player_gives_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> None:
        raise OSError("PortAudio library not found")

    monkeypatch.setattr(playback, "SoundDevicePlayer", broken)
    monkeypatch.setattr(playback.shutil, "which", lambda _: None)
    with pytest.raises(PlaybackUnavailableError, match="PortAudio"):
        default_player()


def test_falls_back_to_command_player(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> None:
        raise ImportError

    monkeypatch.setattr(playback, "SoundDevicePlayer", broken)
    monkeypatch.setattr(playback.shutil, "which", lambda name: name == "aplay")
    player = default_player()
    assert isinstance(player, CommandPlayer)
    assert player.command[0] == "aplay"
