"""Calibrate the hand-set parameters on the Metrica open games (never on BAR-ATH).

Objective: mean spellF1 (the official metric code) over the two sample games, with the
ball removed in long blocks to resemble the automatic tracking. Random search, seeded.
"""
import copy
import json
import os
import sys

import numpy as np

from . import features, emissions, decode, metrica, gaps, deadball, setpieces
from .data import ROOT, CACHE_DIR

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

SEARCH = {  # (low, high) sampled uniformly
    "k_team": (0.1, 1.5),
    "control_decay": (0.5, 3.0),
    "flight_speed": (6.0, 16.0),
    "slow_center": (0.8, 1.8),
    "k_slow": (0.05, 0.8),
    "k_still": (0.0, 0.8),
    "d_bias": (-0.6, 0.1),
    "k_dead": (0.0, 0.8),       # k_gap not searched: the dead-ball model already sees the gaps
    "k_sp_dead": (0.0, 1.0),    # set-piece anchors, calibrated with synthetic tags (metrica.py)
    "k_sp_team": (0.0, 1.5),
    "sp_back_scale": (0.5, 1.5),
}
DECODE_SEARCH = {"switch": (2.0, 15.0), "min_pos": (5, 75), "min_dead": (10, 150)}


def prepare(games=(1, 2), seed=0, ball_mode="atd_like", gap_model=False, dead_model=True):
    """[(G, f, gt, gap_logit, dead_logit, synthetic_tags)] per game.

    ball_mode: "atd_like" (default) fills the native Metrica gaps and cuts long random gaps, so a
    missing ball is no longer a proxy for a dead ball (as in the automatic tracking);
    "legacy" is the v1 protocol (native gaps + extra random gaps, 55% missing).
    gap_model: the gap classifier for each game is trained on the OTHER game (leave-one-game-out),
    so the calibration never sees a classifier that was fitted on the game it is scored on.
    dead_model: same leave-one-game-out scheme for the dead-ball classifier (deadball.py).
    """
    rng = np.random.default_rng(seed)
    out = []
    for g in games:
        G = metrica.load_game(g)
        gt = metrica.labels_from_events(G)
        if ball_mode == "atd_like":
            b = metrica.atd_like_ball(G["ball"], G["period"], rng)
        else:
            b = metrica.drop_ball_like_atd(G["ball"], rng, target=0.55)
        G = dict(G, ball=b)
        out.append((G, features.compute(G), gt))
    gl = [np.zeros(G["n_frames"], np.float32) for G, _, _ in out]
    if gap_model and len(out) > 1:
        sets = gaps.training_set(out)
        for i, (G, f, gt) in enumerate(out):
            X = np.concatenate([s[0] for j, s in enumerate(sets) if j != i])
            y = np.concatenate([s[1] for j, s in enumerate(sets) if j != i])
            gl[i] = gaps.gap_logit(f, G["ball"], blocks_of(gt), gaps.fit(X, y))
    dl = [np.zeros(G["n_frames"], np.float32) for G, _, _ in out]
    if dead_model and len(out) > 1:
        sets = deadball.training_set(out)
        for i, (G, f, gt) in enumerate(out):
            X = np.concatenate([s[0] for j, s in enumerate(sets) if j != i])
            y = np.concatenate([s[1] for j, s in enumerate(sets) if j != i])
            dl[i] = deadball.dead_logit(f, G, deadball.fit(X, y))
    tags = [metrica.synthetic_tags(G, rng) for G, _, _ in out]
    return [(G, f, gt, l, d, t) for (G, f, gt), l, d, t in zip(out, gl, dl, tags)]


def fit_gap_model(prepared):
    """Final gap classifier, fitted on ALL the prepared Metrica games (for the challenge game)."""
    sets = gaps.training_set(prepared)
    return gaps.fit(np.concatenate([s[0] for s in sets]), np.concatenate([s[1] for s in sets]))


def blocks_of(gt):
    codes = pm.label_codes(gt)
    return [(0, a, b - 1) for a, b in pm.valid_runs(codes)]


def fit_dead_model(prepared):
    """Final dead-ball classifier, fitted on ALL the prepared Metrica games."""
    sets = deadball.training_set(prepared)
    return deadball.fit(np.concatenate([s[0] for s in sets]), np.concatenate([s[1] for s in sets]))


def score(prepared, params, cfg):
    s = []
    for G, f, gt, gl, dl, tags in prepared:
        ll = (emissions.tracking_scores(f, params) + emissions.gap_scores(gl, params)
              + emissions.dead_scores(dl, params)
              + setpieces.setpiece_scores(tags, f, G["ball"], G["n_frames"], params)[0])
        pred = decode.decode_match(ll, blocks_of(gt), G["n_frames"], cfg)
        gtc, prc = pm.label_codes(gt), pm.label_codes(pred)
        s.append(pm.spell_scores(gtc, prc, pm.valid_runs(gtc))["mean_f1"])
    return float(np.mean(s)), s


def make_cfg(sw, mp, md):
    cfg = copy.deepcopy(decode.DECODE)
    cfg["min_len"] = {emissions.H: int(mp), emissions.A: int(mp), emissions.D: int(md)}
    cfg["switch_cost"] = {k: float(sw) for k in cfg["switch_cost"]}
    return cfg


def sample(rng):
    p = dict(emissions.PARAMS)
    for k, (lo, hi) in SEARCH.items():
        p[k] = float(rng.uniform(lo, hi))
    d = {k: float(rng.uniform(*v)) for k, v in DECODE_SEARCH.items()}
    return p, d


def random_search(n_trials, seed, out_path=None, prepared=None):
    prepared = prepared or prepare()
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_trials):
        p, d = sample(rng)
        m, per = score(prepared, p, make_cfg(d["switch"], d["min_pos"], d["min_dead"]))
        results.append({"spell": m, "per_game": per, "params": p, "decode": d})
        if out_path:
            with open(out_path, "a") as fh:
                fh.write(json.dumps(results[-1]) + "\n")
    return results


if __name__ == "__main__":
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    random_search(n, seed, os.path.join(CACHE_DIR, "calibration.jsonl"))


# --- parallel search with checkpoint/resume --------------------------------------------------
# Every finished trial is appended to a JSONL log (on Google Drive when running on Colab), so a
# runtime reset loses at most the trials in flight; rerunning the same call resumes from the log.
# Trial i of seed s always samples the same configuration (rng seeded with [s, i]).

_PREPARED = None


def _init_worker():
    global _PREPARED
    _PREPARED = prepare()          # once per process, not once per trial


def _eval_trial(task):
    seed, i = task
    p, d = sample(np.random.default_rng([seed, i]))
    m, per = score(_PREPARED, p, make_cfg(d["switch"], d["min_pos"], d["min_dead"]))
    return {"seed": seed, "trial": i, "spell": m, "per_game": per, "params": p, "decode": d}


def load_log(path):
    """Records in a JSONL log; a truncated last line (runtime killed mid-write) is skipped."""
    out = []
    if path and os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def parallel_search(n_trials, n_jobs, seed, log_path=None):
    """Random search over processes. With log_path: resumes from the log and returns ALL the
    records in it (keep one log per code version: old records are scored with old code)."""
    from multiprocessing import Pool
    done = load_log(log_path)
    have = {r["trial"] for r in done if r.get("seed") == seed and "trial" in r}
    todo = [(seed, i) for i in range(n_trials) if i not in have]
    print(f"log: {len(done)} configurazioni gia' fatte, {len(todo)} da fare")
    if todo:
        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with Pool(max(1, min(n_jobs, len(todo))), initializer=_init_worker) as pool:
            for k, r in enumerate(pool.imap_unordered(_eval_trial, todo), 1):
                done.append(r)
                if log_path:
                    with open(log_path, "a") as fh:
                        fh.write(json.dumps(r) + "\n")
                        fh.flush()
                        os.fsync(fh.fileno())
                if k % 10 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)}  migliore finora (max-min): {max(min(x['per_game']) for x in done):.3f}",
                          flush=True)
    return done


def best_of(results):
    """Robust choice: the configuration whose WORSE game is best (avoids a lucky single game)."""
    return max(results, key=lambda r: min(r["per_game"]))


def write_params(best, path, n_trials, seed):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"source": "random search on Metrica Sample Games 1-2", "seed": seed,
                   "n_trials": n_trials, "spell_f1_metrica": best["per_game"],
                   "params": best["params"], "decode": best["decode"]}, fh, indent=2)
    return path
