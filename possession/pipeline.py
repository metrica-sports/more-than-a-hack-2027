"""End to end: data bundle in, 199,319 labels out. Deterministic, no randomness."""

import numpy as np

from games import GAME, GAMES, half_video_blocks
from . import config
from .deadball import dead_probability
from .features import interpolate_ball, team_distances
from .states import A, D, H, LETTERS, viterbi
from .tracking import load_tracking


def log_emissions(d_home, d_away, confidence, p_dead):
    """(N, 3) log-evidence for H, A, D from distances to the ball and the dead-ball prior."""
    diff = np.nan_to_num(d_away - d_home, nan=0.0)                  # >0: home is closer
    vote = 1.0 / (1.0 + np.exp(-diff / config.DISTANCE_SCALE * confidence))
    eps = 1e-6
    alive = 1.0 - p_dead
    out = np.empty((len(diff), 3), dtype=np.float64)
    out[:, H] = np.log(alive * vote + eps)
    out[:, A] = np.log(alive * (1.0 - vote) + eps)
    out[:, D] = np.log(p_dead + eps)
    return out


def predict(ball, players, team, game=GAME):
    """Labels for every frame: viterbi inside the graded halves, `X` elsewhere."""
    n = GAMES[game]["n_video_frames"]
    filled, confidence = interpolate_ball(ball, config.MAX_BALL_GAP)
    d_home, d_away = team_distances(filled, players, team)
    emission = log_emissions(d_home, d_away, confidence,
                             dead_probability(n, confidence, game))

    labels = np.full(n, "X", dtype="<U1")
    for _, first, last in half_video_blocks(game):
        path = viterbi(emission[first:last + 1], config.SWITCH_HOME_AWAY, config.SWITCH_DEAD)
        labels[first:last + 1] = LETTERS[path]
    return labels


def write_csv(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def run(out_path, game=GAME, data_dir=None):
    ball, players, team = load_tracking(game, data_dir)
    labels = predict(ball, players, team, game)
    write_csv(out_path, labels)
    return labels
