"""End-to-end: data -> features -> scores -> decoding -> labels."""
import copy
import json
import os
import warnings

import numpy as np

from . import data, features, emissions, decode, gaps

PARAMS_PATH = os.path.join(data.ROOT, "solution", "params.json")


def params_path():
    """MTAH_PARAMS overrides the default location (e.g. a file on Google Drive)."""
    return os.environ.get("MTAH_PARAMS") or PARAMS_PATH


def gap_model_path():
    """gap_model.json sits next to params.json (on Drive when running on Colab)."""
    return os.environ.get("MTAH_GAP_MODEL") or os.path.join(os.path.dirname(params_path()),
                                                             "gap_model.json")


def load_params(path=None, required=None):
    """Emission + decoding parameters. params.json is written by notebook 02 from the
    calibration on the Metrica open games.

    required=True (or env MTAH_REQUIRE_PARAMS=1): a missing params.json is an error.
    Otherwise the hand-set defaults are used, with a visible warning -- never silently.
    """
    path = path or params_path()
    if required is None:
        required = os.environ.get("MTAH_REQUIRE_PARAMS", "0") == "1"
    params, cfg = dict(emissions.PARAMS), copy.deepcopy(decode.DECODE)
    if not os.path.exists(path):
        msg = (f"params.json non trovato in {path}: esegui prima il notebook 02 "
               "(calibrazione) o indica il file con MTAH_PARAMS")
        if required:
            raise FileNotFoundError(msg)
        warnings.warn(msg + " -- uso i parametri di DEFAULT", stacklevel=2)
        return params, cfg
    with open(path) as fh:
        js = json.load(fh)
    params.update(js["params"])
    d = js["decode"]
    cfg["min_len"] = {emissions.H: int(d["min_pos"]), emissions.A: int(d["min_pos"]),
                      emissions.D: int(d["min_dead"])}  # same truncation as calibrate.make_cfg
    cfg["switch_cost"] = {k: float(d["switch"]) for k in cfg["switch_cost"]}
    return params, cfg


def run(params=None, decode_cfg=None, use_tagging=True, verbose=True, required=None):
    if params is None or decode_cfg is None:
        p0, c0 = load_params(required=required)
        params = p0 if params is None else params
        decode_cfg = c0 if decode_cfg is None else decode_cfg
    atd = data.load_atd()
    f = features.compute(atd)
    ll = emissions.tracking_scores(f, params)
    gap_model = gaps.load(gap_model_path()) if params.get("k_gap", 0) > 0 else None
    if params.get("k_gap", 0) > 0 and gap_model is None:
        raise FileNotFoundError("k_gap > 0 ma gap_model.json manca: esegui il notebook 02 "
                                "(lo scrive accanto a params.json) o indica MTAH_GAP_MODEL")
    gl = gaps.gap_logit(f, atd["ball"], data.graded_blocks(), gap_model)
    ll = ll + emissions.gap_scores(gl, params)
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
    return {"labels": labels, "features": f, "scores": ll, "tag_signs": signs, "gap_logit": gl, "atd": atd,
            "params": params, "decode": decode_cfg}


def write_submission(labels, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{lab}\n" for i, lab in enumerate(labels))
    return path
