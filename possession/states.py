"""Turn noisy per-frame evidence into long, stable segments (Viterbi with switch costs)."""

import numpy as np

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


def viterbi(log_emission, switch_home_away, switch_dead):
    """Most likely H/A/D path. log_emission is (N, 3); returns (N,) ints.

    Staying in a state is free; H<->A costs `switch_home_away`; entering or leaving D costs
    `switch_dead`. Large costs mean few, long possessions -- the thing spellF1 rewards.
    """
    n = len(log_emission)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    cost = np.array([[0, switch_home_away, switch_dead],
                     [switch_home_away, 0, switch_dead],
                     [switch_dead, switch_dead, 0]], dtype=np.float64)

    score = log_emission[0].astype(np.float64)
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        candidate = score[:, None] - cost          # [from, to]
        back[t] = np.argmax(candidate, axis=0)
        score = candidate[back[t], np.arange(3)] + log_emission[t]

    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
