"""Smart Tagging set pieces (and goals) as anchors for the HMM.

A SET PIECES instance of the Smart Tagging is a short window (a few seconds) around a restart, with
a Type (Throw-in, Corner Kick, Goal Kick, Free Kick, Kick Off (After Goal), Kick Off (Start)) and
usually the Team (N/A for every throw-in). Each one tells us three things:

1. there IS a restart in (or close to) that window -> the frame where the ball starts moving again
   is found in the tracking (`restart_frame`): ball still, then a burst, ideally at the spot the
   Type implies (touchline, corner flag, 6-yard box, centre spot);
2. before it the ball was dead for a type-dependent time (median D before the restart in the
   Metrica games: throw-in ~7 s, free kick ~10-15 s, goal kick ~20 s, corner ~28 s, kickoff after a
   goal ~55 s) -> D evidence on a ramp ending at the restart;
3. right after it the team that takes the restart has the ball -> team evidence, and a clean
   possession boundary.

GOALS: from the end of a GOALS phase to the next kickoff restart the ball is dead.

The Smart Tagging is automatic and misses many restarts, so this is evidence for the HMM, never a
label. Weights (k_sp_dead, k_sp_team, sp_back_scale) are calibrated on the Metrica games with
synthetic tags built from their events (metrica.synthetic_tags), never on the challenge game.
"""
import numpy as np

from .data import FPS, PITCH, HOME, AWAY
from .emissions import H, A, D, TEAM_TAG

# median seconds of dead ball before the restart (Metrica Sample Games 1-2, labels from events)
BACK_S = {"throw-in": 7.0, "corner kick": 28.0, "goal kick": 20.0, "free kick": 12.0,
          "kick off (after goal)": 55.0, "kick off (start)": 0.0, None: 8.0}
SEARCH_PAD_S = 3.0          # look for the restart this far outside the tag window
AFTER_S = 3.0               # team evidence after the restart
SPOT_M = {"throw-in": 4.0, "corner kick": 4.0, "goal kick": 5.0,
          "kick off (after goal)": 4.0, "kick off (start)": 4.0}


def _type(t):
    v = (t["tags"].get("Type") or "").strip().lower()
    return v if v in BACK_S else None


def _spot_distance(ball, typ):
    """Distance (m) of the ball from where a restart of this type is taken; NaN if unknown."""
    L, W = PITCH
    x, y = ball[:, 0], ball[:, 1]
    if typ == "throw-in":
        return np.minimum(np.abs(y), np.abs(W - y))
    if typ == "corner kick":
        c = np.array([[0, 0], [0, W], [L, 0], [L, W]], np.float32)
        return np.sqrt(((ball[:, None, :] - c[None]) ** 2).sum(-1)).min(1)
    if typ == "goal kick":
        gx = np.minimum(x, L - x)
        return np.sqrt((gx - 5.5) ** 2 + np.maximum(np.abs(y - W / 2) - 9.16, 0) ** 2)
    if typ in ("kick off (after goal)", "kick off (start)"):
        return np.sqrt((x - L / 2) ** 2 + (y - W / 2) ** 2)
    return np.full(len(ball), np.nan, np.float32)


def restart_frame(t, f, ball, n):
    """Most likely restart frame for a tag: the ball goes from still to moving, at the right spot.
    Falls back on the players speeding up, then on the window centre."""
    a = max(int(t["start_frame"] - SEARCH_PAD_S * FPS), 1 * FPS)
    b = min(int(t["end_frame"] + SEARCH_PAD_S * FPS), n - 1 * FPS)
    if b <= a:
        return int(np.clip((t["start_frame"] + t["end_frame"]) // 2, 0, n - 1))
    bs = np.nan_to_num(f["ball_speed"], nan=0.0)
    idx = np.arange(a, b)
    win = FPS                                            # 1 s before / after
    before = np.array([np.nanmean(f["ball_speed"][i - win:i]) for i in idx])
    after = np.array([bs[i:i + win].max() for i in idx])
    seen = f["ball_present"][idx]
    score = np.log1p(after) - np.log1p(np.nan_to_num(before, nan=2.0))
    typ = _type(t)
    d = _spot_distance(ball[idx], typ)
    if typ in SPOT_M:
        score = score + np.where(np.isfinite(d) & (d < SPOT_M[typ]), 0.7, 0.0)
    score = np.where(seen & (after > 3.0), score, -np.inf)
    if np.isfinite(score).any():
        return int(idx[int(np.argmax(score))])
    ps = f["players_speed"]                              # ball not tracked: players restart
    acc = np.array([np.nanmean(ps[i:i + 2 * win]) - np.nanmean(ps[i - 2 * win:i]) for i in idx])
    if np.isfinite(acc).any():
        return int(idx[int(np.nanargmax(acc))])
    return int((t["start_frame"] + t["end_frame"]) // 2)


def _nearest_team(f, i):
    dh, da = f["d_home"][i], f["d_away"][i]
    if not (f["ball_present"][i] and np.isfinite(dh) and np.isfinite(da)):
        return None
    return HOME if dh <= da else AWAY


def _taker(f, r):
    """Team nearest to the ball in the second after the restart (majority)."""
    votes = [_nearest_team(f, i) for i in range(r, min(r + FPS, len(f["d_home"])))]
    votes = [v for v in votes if v is not None]
    if not votes:
        return None
    return HOME if votes.count(HOME) >= votes.count(AWAY) else AWAY


def anchors(tags, f, ball, n):
    """[(type, restart_frame, taker_team or None)] for the SET PIECES tags, and the GOALS windows.
    The meaning of the Team tag (taker or opponent) is decided algorithmically by agreement with
    the nearest player right after the detected restarts (no ground truth involved)."""
    sp = [t for t in tags if t["code"] == "SET PIECES"]
    rs = [restart_frame(t, f, ball, n) for t in sp]
    agree = total = 0
    for t, r in zip(sp, rs):
        tag_team = TEAM_TAG.get((t["tags"].get("Team") or "").strip().lower())
        near = _taker(f, r)
        if tag_team is not None and near is not None:
            agree += int(tag_team == near)
            total += 1
    sign = 1 if total == 0 or agree / total >= 0.5 else -1
    out = []
    for t, r in zip(sp, rs):
        tag_team = TEAM_TAG.get((t["tags"].get("Team") or "").strip().lower())
        team = (tag_team if sign > 0 else 1 - tag_team) if tag_team is not None else _taker(f, r)
        out.append((_type(t), r, team))
    goals = [(int(t["start_frame"]), int(t["end_frame"])) for t in tags if t["code"] == "GOALS"]
    return out, goals, {"team_sign": sign, "team_agreement": (agree, total)}


def setpiece_scores(tags, f, ball, n, p):
    """(n, 3) evidence from set pieces and goals; zero when the weights are zero."""
    ll = np.zeros((n, 3), np.float32)
    kd, kt, scale = p.get("k_sp_dead", 0.0), p.get("k_sp_team", 0.0), p.get("sp_back_scale", 1.0)
    if kd == 0 and kt == 0:
        return ll, None
    sps, goals, info = anchors(tags, f, ball, n)
    for typ, r, team in sps:
        back = int(BACK_S.get(typ, BACK_S[None]) * scale * FPS)
        if back > 0 and kd > 0:
            a = max(r - back, 0)
            ramp = np.linspace(0.3, 1.0, r - a, dtype=np.float32)   # surest near the restart
            ll[a:r, D] += kd * ramp
        e = min(r + int(AFTER_S * FPS), n)
        if kt > 0:
            ll[r:e, D] -= kt                                       # play is live again
            if team is not None:
                tgt = H if team == HOME else A
                ll[r:e, tgt] += kt
                ll[r:e, A if tgt == H else H] -= kt
    kickoffs = sorted(r for typ, r, _ in sps if typ and typ.startswith("kick off"))
    for _, ge in goals:
        nxt = [r for r in kickoffs if ge < r <= ge + 180 * FPS]
        if nxt and kd > 0:
            ll[ge:nxt[0], D] += kd
    return ll, info
