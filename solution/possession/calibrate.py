"""Calibrate the hand-set parameters on the Metrica open games (never on BAR-ATH).

Objective: mean spellF1 (the official metric code) over the two sample games, with the
ball removed in long blocks to resemble the automatic tracking. Random search, seeded.
"""
import copy
import json
import os
import sys

import numpy as np

from . import features, emissions, decode, metrica
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
}
DECODE_SEARCH = {"switch": (2.0, 15.0), "min_pos": (5, 75), "min_dead": (10, 150)}


def prepare(games=(1, 2), seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for g in games:
        G = metrica.load_game(g)
        gt = metrica.labels_from_events(G)
        G = dict(G, ball=metrica.drop_ball_like_atd(G["ball"], rng, target=0.55))
        f = features.compute(G)
        out.append((G, f, gt))
    return out


def blocks_of(gt):
    codes = pm.label_codes(gt)
    return [(0, a, b - 1) for a, b in pm.valid_runs(codes)]


def score(prepared, params, cfg):
    s = []
    for G, f, gt in prepared:
        ll = emissions.tracking_scores(f, params)
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


def _worker(args):
    n, seed = args
    return random_search(n, seed)


def parallel_search(n_trials, n_jobs, seed):
    """Random search split over processes (module-level worker: works with spawn and fork)."""
    from multiprocessing import Pool
    per = max(n_trials // n_jobs, 1)
    with Pool(n_jobs) as pool:
        chunks = pool.map(_worker, [(per, seed + i) for i in range(n_jobs)])
    return [r for c in chunks for r in c]
