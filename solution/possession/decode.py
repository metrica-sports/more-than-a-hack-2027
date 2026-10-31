"""Minimum-duration HMM over the states H, A, D, decoded with Viterbi.

Each state is expanded into a chain of `min_len` sub-states, so a state can only
be left after it has lasted at least min_len frames: a possession of 3 frames is
impossible by construction, which is what the segment metric (spellF1) rewards.
Leaving a state costs a fixed log-penalty.
"""
import numpy as np

from .emissions import H, A, D, STATES

DECODE = {
    "min_len": {H: 25, A: 25, D: 50},     # frames: 1 s possession, 2 s stoppage
    "switch_cost": {                      # log-penalty for (from, to)
        (H, A): 6.0, (A, H): 6.0,
        (H, D): 6.0, (A, D): 6.0,
        (D, H): 6.0, (D, A): 6.0,
    },
}


def _build(cfg):
    """Sub-state layout and the list of allowed predecessor edges."""
    offs, owner = {}, []
    for s in (H, A, D):
        offs[s] = len(owner)
        owner += [s] * cfg["min_len"][s]
    owner = np.array(owner)
    K = len(owner)
    # predecessors as dense matrix of log-transition scores (-inf = not allowed)
    T = np.full((K, K), -np.inf, np.float32)
    for s in (H, A, D):
        o, m = offs[s], cfg["min_len"][s]
        for k in range(m - 1):
            T[o + k, o + k + 1] = 0.0          # forced progress through the chain
        T[o + m - 1, o + m - 1] = 0.0          # stay once the minimum is reached
        for t in (H, A, D):
            if t != s:
                T[o + m - 1, offs[t]] = -cfg["switch_cost"][(s, t)]
    return owner, offs, T


def viterbi(ll, cfg=DECODE, start=None):
    """ll: (n, 3) per-frame scores. Returns state index per frame."""
    owner, offs, T = _build(cfg)
    K, n = len(owner), len(ll)
    em = ll[:, owner]                                   # (n, K)
    delta = np.full(K, -np.inf, np.float32)
    starts = [start] if start is not None else [H, A, D]
    for s in starts:
        delta[offs[s]] = 0.0
    delta = delta + em[0]
    back = np.empty((n, K), np.int16)
    back[0] = np.arange(K)
    for t in range(1, n):
        cand = delta[:, None] + T                       # (K_from, K_to)
        back[t] = np.argmax(cand, 0)
        delta = cand[back[t], np.arange(K)] + em[t]
    path = np.empty(n, np.int64)
    k = int(np.argmax(delta))
    for t in range(n - 1, -1, -1):
        path[t] = k
        k = back[t, k]
    return owner[path]


def decode_match(ll, blocks, n, cfg=DECODE):
    """Decode each graded half independently; X outside the graded frames."""
    labels = np.full(n, "X", dtype="<U1")
    for _, a, b in blocks:
        labels[a:b + 1] = STATES[viterbi(ll[a:b + 1], cfg)]
    return labels
