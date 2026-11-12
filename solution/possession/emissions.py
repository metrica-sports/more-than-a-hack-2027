"""Per-frame log-likelihood scores for the three states H, A, D.

Hand-set, physically motivated parameters (NOT tuned on the public ground truth).
Each frame contributes a small amount of evidence; the HMM in decode.py
accumulates it over time, so a single noisy frame cannot flip the label.
"""
import numpy as np

from .data import HOME, AWAY

H, A, D = 0, 1, 2
STATES = np.array(["H", "A", "D"])

PARAMS = {
    # possession: who is close enough to the ball to control it
    "control_radius": 1.0,      # m: within this distance control evidence is maximal
    "control_decay": 1.5,       # m: e-folding distance beyond the radius
    "flight_speed": 10.0,       # m/s: faster ball = in flight, nobody controls it
    "flight_weight": 0.3,
    "k_team": 0.4,              # evidence per frame for full control vs none
    # dead ball
    "outside_margin": 0.5,      # m beyond the line before calling the ball out
    "k_outside": 1.5,
    "slow_center": 1.2,         # m/s: collective speed below this suggests a stoppage
    "slow_scale": 0.35,
    "k_slow": 0.35,
    "k_still": 0.3,             # ball not moving for ~1 s with slow players
    "d_bias": -0.15,            # per-frame prior against D (play is live most of the time)
    # ball gaps: weight of the gap classifier's log-odds of a dead ball (gaps.py); 0 = off
    "k_gap": 0.0,
    # dead-ball classifier on every frame (deadball.py): weight of its log-odds; 0 = off
    "k_dead": 0.0,
    # smart tagging
    "k_tag_team": 0.08,
    "k_tag_setpiece": 0.0,      # old SET PIECES prior, replaced by setpieces.py (k_sp_*)
    # set-piece anchors (setpieces.py): D before the restart, taker's team after it
    "k_sp_dead": 0.0,
    "k_sp_team": 0.0,
    "sp_back_scale": 1.0,
}

TEAM_TAG = {"barcelona": HOME, "atletic club": AWAY, "athletic club": AWAY,
            "home": HOME, "away": AWAY}      # home/away: synthetic tags on the Metrica games


def _control(d, p):
    c = np.exp(-np.maximum(d - p["control_radius"], 0) / p["control_decay"])
    return np.where(np.isnan(d), 0.0, c)


def tracking_scores(f, p=PARAMS):
    n = len(f["ball_present"])
    ll = np.zeros((n, 3), np.float32)

    # --- team evidence (only where the ball is seen) ---
    ch, ca = _control(f["d_home"], p), _control(f["d_away"], p)
    w = np.where(f["ball_speed"] > p["flight_speed"], p["flight_weight"], 1.0)
    s = np.where(f["ball_present"], (ch - ca) * w, 0.0)
    ll[:, H] += p["k_team"] * s / 2
    ll[:, A] -= p["k_team"] * s / 2

    # --- dead-ball evidence ---
    out = np.where(f["ball_present"], f["ball_outside"] - p["outside_margin"], -1)
    ll[:, D] += p["k_outside"] * np.clip(out, -1, 1) * (out > 0)
    slow = np.tanh((p["slow_center"] - np.nan_to_num(f["players_speed"], nan=p["slow_center"]))
                   / p["slow_scale"])
    ll[:, D] += p["k_slow"] * slow
    still = np.nan_to_num(f["ball_still"]) * (slow > 0)
    ll[:, D] += p["k_still"] * still
    ll[:, D] += p["d_bias"]
    return ll


def gap_scores(gap_logit, p=PARAMS):
    """Evidence for D inside the ball gaps from the gap classifier (gaps.gap_logit)."""
    ll = np.zeros((len(gap_logit), 3), np.float32)
    ll[:, D] = p.get("k_gap", 0.0) * gap_logit
    return ll


def dead_scores(dead_logit, p=PARAMS):
    """Evidence for D on every frame from the dead-ball classifier (deadball.dead_logit)."""
    ll = np.zeros((len(dead_logit), 3), np.float32)
    ll[:, D] = p.get("k_dead", 0.0) * dead_logit
    return ll


# handled by setpieces.py: their team tag refers to the restart, not to the whole window
ANCHOR_CODES = {"SET PIECES", "GOALS"}


def tag_team_signs(tags, f):
    """For each team-tagged code: +1 if the tag names the team in possession, -1 if the
    opponent. Decided algorithmically by agreement with the nearest-player team on frames
    where someone is clearly on the ball (no ground truth involved)."""
    clear = f["ball_present"] & (np.fmin(f["d_home"], f["d_away"]) < 1.5)
    near = np.where(f["d_home"] < f["d_away"], HOME, AWAY)
    agree, total = {}, {}
    for t in tags:
        team = TEAM_TAG.get((t["tags"].get("Team") or "").strip().lower())
        if team is None or t["code"] in ANCHOR_CODES:
            continue
        sl = slice(t["start_frame"], t["end_frame"])
        m = clear[sl]
        agree[t["code"]] = agree.get(t["code"], 0) + int((near[sl][m] == team).sum())
        total[t["code"]] = total.get(t["code"], 0) + int(m.sum())
    return {c: (1 if agree[c] / max(total[c], 1) >= 0.5 else -1, agree[c] / max(total[c], 1))
            for c in total}


def tagging_scores(tags, f, n, p=PARAMS, signs=None):
    ll = np.zeros((n, 3), np.float32)
    signs = signs if signs is not None else tag_team_signs(tags, f)
    for t in tags:
        a, b = max(t["start_frame"], 0), min(t["end_frame"], n)
        team = TEAM_TAG.get((t["tags"].get("Team") or "").strip().lower())
        if team is not None and t["code"] in signs and t["code"] not in ANCHOR_CODES:
            sign = signs[t["code"]][0]
            tgt = team if sign > 0 else 1 - team
            ll[a:b, tgt] += p["k_tag_team"]
            ll[a:b, 1 - tgt] -= p["k_tag_team"]
        if t["code"] == "SET PIECES" and p["k_tag_setpiece"] > 0:
            # the tag is a ~5 s window around the restart: the first part is still dead
            mid = (a + b) // 2
            ll[a:mid, D] += p["k_tag_setpiece"]
    return ll, signs
