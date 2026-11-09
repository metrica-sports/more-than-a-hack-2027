"""Dead-ball classifier on EVERY frame (ball tracked or not).

Why: in the automatic tracking a stoppage is often still tracked (a player holds the ball for a
throw-in, places it for a free kick). The nearest-player evidence then says "H" or "A" loud and
clear, and the HMM never goes to D. The restart patterns of Vidal-Codina et al. 2022
(arXiv 2202.00804) are the cue: before a restart the ball is still at a typical spot (on the
touchline = throw-in, at a corner flag = corner, on the 6-yard line = goal kick, on the centre spot
with both teams in their own half = kickoff, anywhere with a wall of slow players = free kick), the
players have slowed down, and right after the restart the ball and the players speed up again.

Features are multi-scale (past / now / future windows), so the model can see the "slow down ->
still -> burst" shape. Trained ONLY on the Metrica open games with labels derived from their
events, with the ball made ATD-like (metrica.atd_like_ball). The Smart Tagging is NOT a feature
(Metrica has no tagging): SET PIECES keeps its own prior in emissions.tagging_scores.

The model is a scikit-learn HistGradientBoostingClassifier saved with pickle (deadball_model.pkl
next to params.json); sklearn is needed at inference too.
"""
import os
import pickle

import numpy as np
import pandas as pd

from .data import FPS, PITCH, HOME, AWAY
from .features import runs

FEATURES = (
    # ball (NaN when not tracked)
    "ball_present", "ball_outside", "ball_speed", "ball_still_1s", "ball_still_3s",
    "ball_speed_max_past2s", "ball_speed_max_next2s",
    "to_touchline", "to_goalline", "to_corner", "to_center", "to_goalarea",
    "near_player", "near_gap",                       # nearest player; |d_home - d_away|
    "ball_dx_3s",                                    # ball displacement over the last 3 s
    # time to/from the ball (gap context)
    "log_since_seen", "log_until_seen",
    # players
    "speed_now", "speed_past5s", "speed_next3s", "speed_min5s", "speed_p90_now",
    "halves_split",                                  # kickoff: each team in its own half
    "spread_x", "players_near_ball10",
)


def _roll(x, w, how="mean", shift=0):
    """Rolling stat over a window of w frames; shift>0 = window in the future, <0 = past,
    0 = centred."""
    s = pd.Series(x)
    if shift == 0:
        r = s.rolling(w, center=True, min_periods=1)
    else:
        r = s.rolling(w, min_periods=1)
    out = getattr(r, how)().to_numpy(np.float32)
    if shift > 0:                       # window [t+1, t+w]
        out = np.concatenate([out[w:], np.full(w, np.nan, np.float32)])
    return out


def _since_last(present):
    """Frames since the ball was last seen (0 when seen) and until it is seen again."""
    n = len(present)
    since = np.zeros(n, np.float32)
    until = np.zeros(n, np.float32)
    for a, b in runs(~present):
        since[a:b] = np.arange(1, b - a + 1)
        until[a:b] = np.arange(b - a, 0, -1)
    return since, until


def frame_features(f, atd):
    """(n, len(FEATURES)) float32 matrix for every frame."""
    ball, players, team = atd["ball"], atd["players"], atd["team"]
    L, W = PITCH
    n = len(ball)
    x, y = ball[:, 0], ball[:, 1]
    present = f["ball_present"]
    bs = f["ball_speed"]
    still = (bs < 1.0).astype(np.float32)
    still[np.isnan(bs)] = np.nan

    # restart spots (metres, pitch origin at a corner)
    to_touch = np.minimum(y, W - y)
    to_goal = np.minimum(x, L - x)
    corners = np.array([[0, 0], [0, W], [L, 0], [L, W]], np.float32)
    to_corner = np.sqrt(((ball[:, None, :] - corners[None]) ** 2).sum(-1)).min(1)
    to_center = np.sqrt((x - L / 2) ** 2 + (y - W / 2) ** 2)
    # goal kick: ball on the 6-yard box (5.5 m from the goal line, within 9.16 m of the axis)
    to_goalarea = np.sqrt(np.maximum(np.abs(to_goal - 5.5), 0) ** 2
                          + np.maximum(np.abs(y - W / 2) - 9.16, 0) ** 2)

    dh, da = f["d_home"], f["d_away"]
    with np.errstate(all="ignore"):
        near = np.fmin(dh, da)
        near_gap = np.abs(dh - da)
    dx3 = np.full(n, np.nan, np.float32)
    k = 3 * FPS
    dx3[k:] = np.sqrt(((ball[k:] - ball[:-k]) ** 2).sum(-1))

    since, until = _since_last(present)

    # players: speeds normalised by the game median (camera / tracker differences)
    ps = f["players_speed"]
    med = np.nanmedian(ps) if np.any(np.isfinite(ps)) else 1.0
    med = med if med > 0 else 1.0
    from .features import _central_speed
    sp = _central_speed(players)
    sp = np.where(sp > 12, np.nan, sp)
    with np.errstate(all="ignore"):
        p90 = np.nanpercentile(np.where(np.isnan(sp), np.nan, sp), 90, axis=1) / med
        px = players[:, :, 0]
        hx = np.nanmean(px[:, team == HOME], 1)
        ax = np.nanmean(px[:, team == AWAY], 1)
        home_left = (hx < ax)[:, None]
        own = np.where(team[None, :] == HOME,
                       np.where(home_left, px < L / 2, px > L / 2),
                       np.where(home_left, px > L / 2, px < L / 2))
        vis = ~np.isnan(px)
        split = (own & vis).sum(1) / np.maximum(vis.sum(1), 1)
        spread = np.nanstd(px, 1)
        dball = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))
        near10 = (dball < 10).sum(1).astype(np.float32)
    near10[~present] = np.nan
    p90 = _roll(p90, FPS, "mean")

    cols = {
        "ball_present": present.astype(np.float32),
        "ball_outside": np.clip(f["ball_outside"], -15, 10),
        "ball_speed": bs,
        "ball_still_1s": _roll(still, FPS, "mean"),
        "ball_still_3s": _roll(still, 3 * FPS, "mean"),
        "ball_speed_max_past2s": _roll(bs, 2 * FPS, "max", shift=-1),
        "ball_speed_max_next2s": _roll(bs, 2 * FPS, "max", shift=1),
        "to_touchline": to_touch, "to_goalline": to_goal, "to_corner": to_corner,
        "to_center": to_center, "to_goalarea": to_goalarea,
        "near_player": np.minimum(near, 20), "near_gap": np.minimum(near_gap, 20),
        "ball_dx_3s": dx3,
        "log_since_seen": np.log1p(since / FPS), "log_until_seen": np.log1p(until / FPS),
        "speed_now": ps / med,
        "speed_past5s": _roll(ps, 5 * FPS, "mean", shift=-1) / med,
        "speed_next3s": _roll(ps, 3 * FPS, "mean", shift=1) / med,
        "speed_min5s": _roll(ps, 5 * FPS, "min") / med,
        "speed_p90_now": p90,
        "halves_split": _roll(split.astype(np.float32), FPS, "mean"),
        "spread_x": _roll(spread.astype(np.float32), FPS, "mean"),
        "players_near_ball10": near10,
    }
    X = np.stack([np.asarray(cols[c], np.float32) for c in FEATURES], 1)
    return X


# --- model ---------------------------------------------------------------------------------------

HGB_KW = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
              l2_regularization=1.0, random_state=0)


def fit(X, y, subsample=4, **kw):
    """Gradient boosting; consecutive frames are near-duplicates, so every `subsample`-th frame
    is enough and 4x faster."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(**dict(HGB_KW, **kw))
    clf.fit(X[::subsample], y[::subsample])
    return {"model": clf, "features": list(FEATURES), "prior": float(np.mean(y))}


def predict_logit(m, X, clip=4.0):
    assert m["features"] == list(FEATURES), "deadball model fatto con altre feature"
    p = np.clip(m["model"].predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    p0 = np.clip(m["prior"], 1e-3, 1 - 1e-3)
    z = np.log(p / (1 - p)) - np.log(p0 / (1 - p0))   # centred: base rate stays with d_bias
    return np.clip(z, -clip, clip).astype(np.float32)


def save(m, path, **meta):
    import sklearn
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(dict(m, sklearn_version=sklearn.__version__, **meta), fh)
    return path


def load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def dead_logit(f, atd, m):
    if m is None:
        return np.zeros(len(atd["ball"]), np.float32)
    return predict_logit(m, frame_features(f, atd))


def training_set(prepared):
    """Per game (X, y) on the frames with a label (gt != X)."""
    out = []
    for G, f, gt, *_ in prepared:
        X = frame_features(f, G)
        keep = gt != "X"
        out.append((X[keep], (gt[keep] == "D").astype(int)))
    return out
