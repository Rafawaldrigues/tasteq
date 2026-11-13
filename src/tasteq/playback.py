"""Local audio playback for the CLI.

Uses ``sounddevice`` when PortAudio is available and otherwise falls back to
a system player (PipeWire, PulseAudio, ALSA or ffmpeg). The Gradio demo plays
audio in the browser and does not need any of this.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

import soundfile as sf

from tasteq.dsp import FloatArray

COMMAND_PLAYERS: tuple[tuple[str, ...], ...] = (
    ("pw-play",),
    ("paplay",),
    ("aplay", "-q"),
    ("ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"),
)


class PlaybackUnavailableError(RuntimeError):
    """No way to play audio on this machine."""


class Player(Protocol):
    """Non-blocking audio output."""

    def start(self, audio: FloatArray, sample_rate: int) -> None:
        """Begin playing; returns immediately."""

    def wait(self) -> None:
        """Block until playback finishes."""

    def stop(self) -> None:
        """Stop playback if it is running."""


class SoundDevicePlayer:
    """Playback through PortAudio via ``sounddevice``."""

    def __init__(self) -> None:
        import sounddevice

        self._sd = sounddevice

    def start(self, audio: FloatArray, sample_rate: int) -> None:
        """Begin playing; returns immediately."""
        self._sd.play(audio.astype("float32"), sample_rate)

    def wait(self) -> None:
        """Block until playback finishes."""
        self._sd.wait()

    def stop(self) -> None:
        """Stop playback if it is running."""
        self._sd.stop()


class CommandPlayer:
    """Playback by writing a temporary WAV and running a system player."""

    def __init__(self, command: tuple[str, ...]) -> None:
        self.command = command
        self._process: subprocess.Popen[bytes] | None = None
        self._dir = tempfile.TemporaryDirectory(prefix="tasteq-")

    def start(self, audio: FloatArray, sample_rate: int) -> None:
        """Begin playing; returns immediately."""
        self.stop()
        path = Path(self._dir.name) / "play.wav"
        sf.write(path, audio, sample_rate, subtype="FLOAT")
        self._process = subprocess.Popen(
            [*self.command, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def wait(self) -> None:
        """Block until playback finishes."""
        if self._process is not None:
            self._process.wait()

    def stop(self) -> None:
        """Stop playback if it is running."""
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            self._process.wait()
        self._process = None


def default_player() -> Player:
    """Best available player, or :class:`PlaybackUnavailableError` with a hint."""
    try:
        return SoundDevicePlayer()
    except (ImportError, OSError):
        pass
    for command in COMMAND_PLAYERS:
        if shutil.which(command[0]):
            return CommandPlayer(command)
    raise PlaybackUnavailableError(
        "no audio output found: install PortAudio (e.g. `sudo apt install libportaudio2`) "
        "or a command-line player (pw-play, paplay, aplay, ffplay), or use `tasteq demo`"
    )
