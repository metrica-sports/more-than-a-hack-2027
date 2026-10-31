"""End-to-end: data -> features -> scores -> decoding -> labels."""
import copy
import json
import os

import numpy as np

from . import data, features, emissions, decode

PARAMS_PATH = os.path.join(data.ROOT, "solution", "params.json")


def load_params(path=PARAMS_PATH):
    """Emission + decoding parameters. params.json is written by notebook 02 from the
    calibration on the Metrica open games; without it the hand-set defaults are used."""
    params, cfg = dict(emissions.PARAMS), copy.deepcopy(decode.DECODE)
    if os.path.exists(path):
        with open(path) as fh:
            js = json.load(fh)
        params.update(js["params"])
        d = js["decode"]
        cfg["min_len"] = {emissions.H: int(d["min_pos"]), emissions.A: int(d["min_pos"]),
                          emissions.D: int(d["min_dead"])}
        cfg["switch_cost"] = {k: float(d["switch"]) for k in cfg["switch_cost"]}
    return params, cfg


def run(params=None, decode_cfg=None, use_tagging=True, verbose=True):
    if params is None or decode_cfg is None:
        p0, c0 = load_params()
        params, decode_cfg = params or p0, decode_cfg or c0
    atd = data.load_atd()
    f = features.compute(atd)
    ll = emissions.tracking_scores(f, params)
    signs = None
    if use_tagging:
        tags = data.load_smart_tagging()
        ll_tag, signs = emissions.tagging_scores(tags, f, atd["n_frames"], params)
        ll = ll + ll_tag
    labels = decode.decode_match(ll, data.graded_blocks(), atd["n_frames"], decode_cfg)
    if verbose:
        g = data.graded_mask()
        u, c = np.unique(labels[g], return_counts=True)
        print("graded label counts:", {str(k): int(v) for k, v in zip(u, c)})
    return {"labels": labels, "features": f, "scores": ll, "tag_signs": signs, "atd": atd,
            "params": params, "decode": decode_cfg}


def write_submission(labels, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{lab}\n" for i, lab in enumerate(labels))
    return path
