"""Listen-and-learn: play whole tracks with the predicted best EQ and keep learning."""

from __future__ import annotations

import numpy as np

from tasteq.calibrate import Prompt, ask_in_terminal
from tasteq.playback import Player
from tasteq.session import Tuner


def run_listening(
    tuner: Tuner,
    player: Player,
    prompt: Prompt,
    explore_every: int = 3,
    max_tracks: int | None = None,
    seed: int = 0,
) -> int:
    """Shuffle the library, play each track with its best EQ and collect feedback.

    During playback: ``+`` 👍, ``-`` 👎, Enter for the next track, ``q`` to quit.
    Every ``explore_every`` tracks, one exploratory A/B question on the excerpt
    keeps the model from settling too early. Returns the number of tracks played.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tuner.library.tracks))
    played = 0
    for index in order:
        if max_tracks is not None and played >= max_tracks:
            break
        track = tuner.library.tracks[int(index)]
        point = tuner.best_point(track)
        gains = tuner.space.to_gains(point)
        print(f"\n♪ {track.analysis.path}")
        print("  EQ " + " ".join(f"{g:+.1f}" for g in gains))
        player.start(tuner.renderer.full(track, gains), tuner.renderer.sample_rate)
        reply = prompt("[+] like  [-] dislike  [Enter] next  [q] quit").strip()
        player.stop()
        if reply in {"+", "-"}:
            tuner.thumb(track, point, liked=reply == "+")
        played += 1
        if reply == "q":
            break
        if explore_every and played % explore_every == 0:
            print("  Quick A/B to keep learning:")
            if ask_in_terminal(tuner, tuner.ask(track), player, prompt) is None:
                break
    return played
