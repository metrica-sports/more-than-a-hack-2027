"""Read the raw ATD text file into arrays indexed by *video* frame (not by ATD frame_id)."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402

HOME, AWAY = 0, 1


def load_tracking(game=GAME, data_dir=None):
    """(ball, players, team) in metres.

    ball    (N, 2)      NaN where the tracker lost it (about 41% of graded frames)
    players (N, S, 2)   NaN where a slot is empty; S = 46 slots, NOT 22 players
    team    (S,)        HOME / AWAY per slot
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    team = team_of_each_track(meta_path)
    slots = len(team)

    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, slots, 2), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # ATD counts from 1, submissions from 0
            if not 0 <= frame < n_frames:
                continue
            for slot, group in enumerate(parts[1].split(";")[:slots]):
                if group[:1] == "N":
                    continue
                x, y = group.split(",")[:2]
                players[frame, slot] = (float(x) * length, float(y) * width)
            if parts[2][:1] != "N":
                x, y = parts[2].split(",")[:2]
                ball[frame] = (float(x) * length, float(y) * width)
    return ball, players, team
