"""Metrica Sports open sample data (OTHER games, public, CC-licensed) used to calibrate
and validate the pipeline. Labels are derived ALGORITHMICALLY from the published event
data -- no human labelling by us, and nothing about the challenge game.

Source: https://github.com/metrica-sports/sample-data  (Sample_Game_1, Sample_Game_2)
"""
import os
import urllib.request

import numpy as np
import pandas as pd

from .data import CACHE_DIR, HOME, AWAY

URL = ("https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
       "Sample_Game_{g}/Sample_Game_{g}_{f}.csv")
FILES = ("RawTrackingData_Home_Team", "RawTrackingData_Away_Team", "RawEventsData")
PITCH = (105.0, 68.0)
DIR = os.path.join(CACHE_DIR, "metrica")


def download(games=(1, 2)):
    os.makedirs(DIR, exist_ok=True)
    for g in games:
        for f in FILES:
            path = os.path.join(DIR, f"Sample_Game_{g}_{f}.csv")
            if not os.path.exists(path):
                urllib.request.urlretrieve(URL.format(g=g, f=f), path)


def _tracking(path):
    raw = pd.read_csv(path, skiprows=2)
    cols = list(raw.columns)
    xy = []
    for i, c in enumerate(cols):
        if c.startswith("Player"):
            xy.append(raw.iloc[:, [i, i + 1]].to_numpy(np.float32))
    bi = cols.index("Ball")
    ball = raw.iloc[:, [bi, bi + 1]].to_numpy(np.float32)
    return raw["Period"].to_numpy(), np.stack(xy, 1), ball


def load_game(g):
    download((g,))
    p = lambda f: os.path.join(DIR, f"Sample_Game_{g}_{f}.csv")
    per, ph, ball = _tracking(p(FILES[0]))
    _, pa, _ = _tracking(p(FILES[1]))
    players = np.concatenate([ph, pa], 1)
    team = np.array([HOME] * ph.shape[1] + [AWAY] * pa.shape[1], np.int8)
    scale = np.array(PITCH, np.float32)
    ev = pd.read_csv(p(FILES[2]))
    return {"ball": ball * scale, "players": players * scale, "team": team,
            "n_frames": len(per), "period": per, "events": ev}


def _is_dead_trigger(row):
    t, s = row["Type"], str(row["Subtype"]).upper()
    if t == "BALL OUT" or t == "FAULT RECEIVED":
        return True
    if t == "SHOT" and ("OUT" in s or "GOAL" in s):
        return True
    if t == "BALL LOST" and ("OFFSIDE" in s or "END HALF" in s):
        return True
    return False


CONTROL = {"PASS", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST"}


def labels_from_events(game):
    """Per-frame H/A/D (X before the first kickoff of each period / after its end).

    Rules: control events set the team from their start frame; a dead trigger (ball out,
    foul, goal, offside) sets D from its frame until the next restart (set piece or goal
    kick), which sets the team again.
    """
    n, per = game["n_frames"], game["period"]
    ev = game["events"].sort_values("Start Frame").reset_index(drop=True)
    lab = np.full(n, "X", dtype="<U1")
    for period in np.unique(per):
        e = ev[ev["Period"] == period]
        frames = np.where(per == period)[0]
        f0, f1 = frames[0], frames[-1] + 1
        state, cursor = None, None
        for _, r in e.iterrows():
            fr = int(r["Start Frame"]) - 1
            team = "H" if r["Team"] == "Home" else "A"
            restart = r["Type"] == "SET PIECE" or "GOAL KICK" in str(r["Subtype"]).upper()
            if state is None:
                if r["Type"] in CONTROL:
                    state, cursor = team, fr
                continue
            if state == "D" and not (restart or r["Type"] in CONTROL):
                continue
            if r["Type"] in CONTROL and (team != state):
                lab[cursor:fr] = state
                state, cursor = team, fr
            if _is_dead_trigger(r):
                end = int(r["End Frame"]) - 1 if r["Type"] in ("BALL OUT", "SHOT") and r["End Frame"] > 0 else fr
                end = max(end, cursor)
                lab[cursor:end] = state
                state, cursor = "D", end
        last = int(e["End Frame"].max())
        lab[cursor:min(last, f1)] = state
    return lab


def fill_native_gaps(ball, period):
    """Linear interpolation of the ball inside each period. In the Metrica samples the ball is
    missing almost only at dead ball (~95% of D frames vs ~10% of live frames), while in the
    automatic tracking a stoppage is often still tracked: filling first and then cutting random
    gaps (atd_like_ball) removes the 'missing = dead' shortcut. During a stoppage the filled ball
    drifts slowly from where it went out to where play restarts, roughly a ball being carried
    back."""
    b = ball.copy()
    for p in np.unique(period):
        sl = np.where(period == p)[0]
        seg = b[sl]
        ok = ~np.isnan(seg[:, 0])
        if ok.sum() < 2:
            continue
        t = np.arange(len(sl))
        for k in range(2):
            seg[:, k] = np.interp(t, t[ok], seg[ok, k])   # edges held constant
        b[sl] = seg
    return b


def atd_like_ball(ball, period, rng, target=0.41, median_gap=268):
    """Ball as in the automatic tracking: native gaps filled, then long random gaps cut."""
    return drop_ball_like_atd(fill_native_gaps(ball, period), rng, target, median_gap)


def drop_ball_like_atd(ball, rng, target=0.41, median_gap=268):
    """Remove the ball in long blocks so the clean sample data resembles the automatic
    tracking (in BAR-ATH the ball is missing ~41% of graded frames, in few long gaps)."""
    b = ball.copy()
    n = len(b)
    missing = np.isnan(b[:, 0])
    while missing.mean() < target:
        length = int(rng.lognormal(np.log(median_gap), 0.6))
        start = int(rng.integers(0, max(n - length, 1)))
        missing[start:start + length] = True
    b[missing] = np.nan
    return b
