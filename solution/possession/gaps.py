"""Ball-gap classifier: inside a stretch where the ball is not tracked, is the ball dead?

In the automatic tracking the ball disappears in FEW LONG gaps (~11 s median) and, unlike the
Metrica sample data, a missing ball says little about a stoppage. Inside a gap the HMM has no
team evidence, so the key decision is "dead ball, or the same possession going on?".

For every frame inside a gap we build context features (what the ball was doing when it
disappeared and when it came back, how fast the players move, how long the gap is) and a
logistic regression gives P(dead). It is trained ONLY on the Metrica open games, with labels
derived from their events (metrica.labels_from_events); nothing about the challenge game.

Training data is made ATD-like by `metrica.atd_like_ball`: the native Metrica gaps (which are
almost all dead balls) are filled, then long random gaps are cut, so "ball missing" is no longer
a proxy for "dead" and the model has to learn from the context.

Inference needs numpy only: the fitted model is a JSON file (solution/gap_model.json).
"""
import json
import os

import numpy as np

from .data import FPS, PITCH, HOME, AWAY, ROOT
from .features import runs

MODEL_PATH = os.path.join(ROOT, "solution", "gap_model.json")
FEATURES = (
    "log_len",          # gap duration (log s)
    "log_edge",         # distance in time from the nearest gap end (log s)
    "rel_pos",          # 0 at the start, 1 at the end of the gap
    "in_outside",       # m beyond the line where the ball was last seen (<0 inside)
    "out_outside",      # same where the ball reappears
    "in_speed",         # ball speed when it disappeared (m/s)
    "out_speed",        # ball speed when it reappears
    "in_near",          # nearest player to the ball when it disappeared (m, capped)
    "out_near",
    "displacement",     # entry -> exit ball distance (m)
    "out_to_center",    # exit near the centre spot = kickoff after a goal
    "out_to_corner",    # exit near a corner flag
    "team_change",      # nearest team at entry != at exit
    "speed_now",        # collective player speed at the frame / game median
    "speed_gap_min",    # slowest moment of the gap / game median
    "speed_gap_mean",
)


def _edge_value(x, idx):
    v = x[idx] if 0 <= idx < len(x) else np.nan
    return v


def _nearest(f, i):
    if not (0 <= i < len(f["d_home"])):
        return np.nan, -1
    dh, da = f["d_home"][i], f["d_away"][i]
    if np.isnan(dh) and np.isnan(da):
        return np.nan, -1
    return (np.nanmin([dh, da]), HOME if np.nan_to_num(dh, nan=1e9) <= np.nan_to_num(da, nan=1e9) else AWAY)


def gap_features(f, ball, blocks, max_len_s=None):
    """(idx, X): indices of the frames inside ball gaps (within the given blocks) and their
    feature matrix, columns in FEATURES order."""
    L, W = PITCH
    n = len(ball)
    ps = f["players_speed"]
    med = np.nanmedian(ps[f["ball_present"]]) if np.any(f["ball_present"]) else np.nanmedian(ps)
    med = med if np.isfinite(med) and med > 0 else 1.0
    corners = np.array([[0, 0], [0, W], [L, 0], [L, W]], np.float32)
    center = np.array([L / 2, W / 2], np.float32)
    all_idx, rows = [], []
    for _, a0, b0 in blocks:
        missing = ~f["ball_present"][a0:b0 + 1]
        for a, b in runs(missing):
            a, b = a + a0, b + a0                    # gap = frames [a, b)
            glen = b - a
            ii, io = a - 1, b                         # last seen / first seen again
            in_ok, out_ok = ii >= a0, io <= b0
            in_out = _edge_value(f["ball_outside"], ii) if in_ok else np.nan
            out_out = _edge_value(f["ball_outside"], io) if out_ok else np.nan
            in_sp = _edge_value(f["ball_speed"], ii) if in_ok else np.nan
            out_sp = _edge_value(f["ball_speed"], io) if out_ok else np.nan
            in_near, in_team = _nearest(f, ii) if in_ok else (np.nan, -1)
            out_near, out_team = _nearest(f, io) if out_ok else (np.nan, -1)
            disp = (np.sqrt(((ball[ii] - ball[io]) ** 2).sum()) if in_ok and out_ok else np.nan)
            if out_ok:
                to_c = np.sqrt(((ball[io] - center) ** 2).sum())
                to_k = np.sqrt(((ball[io] - corners) ** 2).sum(1)).min()
            else:
                to_c = to_k = np.nan
            tch = float(in_team != out_team) if in_team >= 0 and out_team >= 0 else 0.5
            seg = ps[a:b] / med
            with np.errstate(all="ignore"):
                smin, smean = np.nanmin(seg) if np.any(np.isfinite(seg)) else np.nan, np.nanmean(seg)
            t = np.arange(a, b)
            edge = np.minimum(t - a + 1, b - t) / FPS
            const = [np.log(glen / FPS), None, None, in_out, out_out, in_sp, out_sp,
                     min(in_near, 15) if np.isfinite(in_near) else np.nan,
                     min(out_near, 15) if np.isfinite(out_near) else np.nan,
                     disp, to_c, to_k, tch, None, smin, smean]
            X = np.empty((glen, len(FEATURES)), np.float32)
            for j, v in enumerate(const):
                if v is not None:
                    X[:, j] = v
            X[:, 1] = np.log(edge)
            X[:, 2] = (t - a + 0.5) / glen
            X[:, 13] = ps[a:b] / med
            all_idx.append(t)
            rows.append(X)
    if not rows:
        return np.zeros(0, int), np.zeros((0, len(FEATURES)), np.float32)
    return np.concatenate(all_idx), np.concatenate(rows)


# --- model: standardise + impute (training median) + logistic regression -----------------------

def fit(X, y, C=0.3, sample_weight=None):
    from sklearn.linear_model import LogisticRegression   # training only
    med = np.nanmedian(X, 0)
    Xi = np.where(np.isnan(X), med, X)
    mu, sd = Xi.mean(0), Xi.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit((Xi - mu) / sd, y, sample_weight=sample_weight)
    return {"features": list(FEATURES), "median": med.tolist(), "mean": mu.tolist(),
            "std": sd.tolist(), "coef": clf.coef_[0].tolist(),
            "intercept": float(clf.intercept_[0]), "prior": float(np.mean(y))}


def predict_logit(model, X):
    assert list(model["features"]) == list(FEATURES), "gap_model.json fatto con altre feature"
    med, mu, sd = (np.asarray(model[k], np.float32) for k in ("median", "mean", "std"))
    Xi = np.where(np.isnan(X), med, X)
    return ((Xi - mu) / sd) @ np.asarray(model["coef"], np.float32) + model["intercept"]


def save(model, path=MODEL_PATH, **meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(dict(model, **meta), fh, indent=2)
    return path


def load(path=None):
    path = path or os.environ.get("MTAH_GAP_MODEL") or MODEL_PATH
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def gap_logit(f, ball, blocks, model, clip=4.0):
    """(n,) array: model logit of 'dead ball' on gap frames (centred on the training prior),
    0 elsewhere. Centring keeps the base rate of D in the hands of d_bias / the HMM."""
    out = np.zeros(len(ball), np.float32)
    if model is None:
        return out
    idx, X = gap_features(f, ball, blocks)
    if len(idx):
        p0 = np.clip(model["prior"], 1e-3, 1 - 1e-3)
        z = predict_logit(model, X) - np.log(p0 / (1 - p0))
        out[idx] = np.clip(z, -clip, clip)
    return out


# --- training on the Metrica open games ----------------------------------------------------------

def training_set(prepared):
    """prepared: list of (G, f, gt) as built by calibrate.prepare. Returns per game (X, y)."""
    from .calibrate import blocks_of
    out = []
    for G, f, gt, *_ in prepared:
        idx, X = gap_features(f, G["ball"], blocks_of(gt))
        keep = gt[idx] != "X"
        out.append((X[keep], (gt[idx][keep] == "D").astype(int)))
    return out
