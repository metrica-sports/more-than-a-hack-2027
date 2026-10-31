"""Per-frame features computed from the tracking only.

All distances in metres, speeds in m/s. NaN means "not observable on this frame".
"""
import numpy as np
import pandas as pd

from .data import FPS, PITCH, HOME, AWAY


def runs(mask):
    """[(start, end_exclusive)] of consecutive True values."""
    m = np.concatenate(([False], np.asarray(mask, bool), [False]))
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _central_speed(xy, h=2):
    """Speed from a centred difference over +-h frames; NaN where either end is missing."""
    v = np.full(xy.shape[:-1], np.nan, np.float32)
    dxy = xy[2 * h:] - xy[:-2 * h]
    v[h:-h] = np.sqrt((dxy ** 2).sum(-1)) * FPS / (2 * h)
    return v


def rolling(x, window, how="mean", min_periods=1):
    s = pd.Series(x)
    r = s.rolling(window, center=True, min_periods=min_periods)
    return getattr(r, how)().to_numpy(np.float32)


def compute(atd):
    ball, players, team = atd["ball"], atd["players"], atd["team"]
    L, W = PITCH
    f = {}
    present = ~np.isnan(ball[:, 0])
    f["ball_present"] = present

    # ball speed, robust to single-frame jumps of the automatic tracker
    bs = _central_speed(ball)
    f["ball_speed"] = rolling(bs, 5, "median")

    # nearest player of each team to the ball
    d = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))          # (N, 46)
    with np.errstate(all="ignore"):
        f["d_home"] = np.where(np.isnan(d[:, team == HOME]).all(1), np.nan,
                               np.nanmin(np.where(np.isnan(d[:, team == HOME]), np.inf, d[:, team == HOME]), 1))
        f["d_away"] = np.where(np.isnan(d[:, team == AWAY]).all(1), np.nan,
                               np.nanmin(np.where(np.isnan(d[:, team == AWAY]), np.inf, d[:, team == AWAY]), 1))

    # how far the ball is outside the pitch (<=0 inside) and from the nearest line
    x, y = ball[:, 0], ball[:, 1]
    out = np.maximum.reduce([-x, x - L, -y, y - W])
    f["ball_outside"] = out.astype(np.float32)
    f["ball_to_line"] = np.abs(out).astype(np.float32)

    # collective player speed: players walk during stoppages
    ps = _central_speed(players)                                        # (N, 46)
    ps = np.where(ps > 12, np.nan, ps)                                  # identity switches
    with np.errstate(all="ignore"):
        mean_speed = np.nanmean(ps, 1)
    f["players_speed"] = rolling(mean_speed, 2 * FPS, "mean")
    f["n_players"] = (~np.isnan(players[:, :, 0])).sum(1)

    # ball stillness over 1 s (placed ball before a restart)
    f["ball_still"] = rolling((f["ball_speed"] < 1.0).astype(np.float32), FPS, "mean")
    return f
