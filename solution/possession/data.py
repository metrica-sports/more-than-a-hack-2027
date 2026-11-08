"""Loading the tactical ATD and the Smart Tagging into numpy arrays.

Only numeric assets are used: tracking (players + ball) and Smart Tagging.
The parsed tracking is cached under solution/cache/ (gitignored: it is derived
match data and must never be committed).
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, half_video_blocks  # noqa: E402

# MTAH_CACHE_DIR lets Colab keep the cache on Google Drive, so it survives runtime resets
# and is shared between notebooks. Must be set before importing this package.
CACHE_DIR = os.environ.get("MTAH_CACHE_DIR") or os.path.join(ROOT, "solution", "cache")
HOME, AWAY = 0, 1
FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]          # (length, width) in metres
N_FRAMES = GAMES[GAME]["n_video_frames"]


def graded_blocks(game=GAME):
    """[(half, first, last)] video frames, inclusive."""
    return half_video_blocks(game)


def graded_mask(n=N_FRAMES, game=GAME):
    m = np.zeros(n, bool)
    for _, a, b in graded_blocks(game):
        m[a:b + 1] = True
    return m


def team_of_slot(meta_path):
    """0 = home, 1 = away for each of the 46 track slots (first <Team> is home)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    return np.array([HOME if p.get("teamId") == team_ids[0] else AWAY
                     for p in root.iter("Player")], dtype=np.int8)


def _parse_raw(raw_path, n_slots, n_frames):
    ball = np.full((n_frames, 2), np.nan, np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, np.float32)
    with open(raw_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1                      # ATD frame_id starts at 1
            b = parts[2].split(",")
            if b[0][0] != "N":
                ball[i] = (float(b[0]), float(b[1]))
            for k, grp in enumerate(parts[1].split(";")):
                if grp[0] != "N":
                    c1 = grp.index(",")
                    c2 = grp.index(",", c1 + 1)
                    players[i, k] = (float(grp[:c1]), float(grp[c1 + 1:c2]))
    return ball, players


def load_atd(game=GAME, data_dir=None, use_cache=True):
    """Dict with ball (N,2) and players (N,46,2) in METRES, team (46,), n_frames.

    Missing observations are NaN. Index = video frame (submission row).
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{game}_tactical_atd.npz")
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_slot(meta_path)
    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        ball, players = z["ball"], z["players"]
    else:
        ball, players = _parse_raw(raw_path, len(team), N_FRAMES)
        np.savez_compressed(cache, ball=ball, players=players)
    scale = np.array(PITCH, np.float32)
    return {"ball": ball * scale, "players": players * scale,
            "team": team, "n_frames": N_FRAMES}


def load_smart_tagging(game=GAME, data_dir=None):
    """List of dicts {code, start_frame, end_frame, tags} (end exclusive)."""
    path = require(["smart_tagging"], game, data_dir)[0]
    out = []
    for inst in ET.parse(path).iter("instance"):
        s, e = float(inst.findtext("start")), float(inst.findtext("end"))
        out.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start_frame": int(round(s * FPS)),
            "end_frame": int(round(e * FPS)),
            "tags": {lab.findtext("group"): lab.findtext("text")
                     for lab in inst.findall("label")},
        })
    return out
