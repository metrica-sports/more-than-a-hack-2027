"""Write a valid submission that answers nothing useful, so you can run the loop before you build.

This exists so that a clone of this repository is immediately runnable with no downloads at all:

    python starter/make_dummy_submission.py
    python evaluation/validate_submission.py evaluation/results/dummy_dead.csv
    python evaluation/run_evaluation.py

Three files, each a different trivial strategy, and each one makes a point worth seeing on the
board before you write any real code:

    dummy_dead.csv      D on every frame. Dead ball is the single most common label, so this is
                        the highest-scoring answer that contains no information. Watch it take
                        a respectable-looking accuracy and a macroF1 near the floor.
    dummy_home.csv      H on every frame.
    dummy_abstain.csv   X on every frame. Scores zero on everything, and is still a legal file --
                        abstaining is always allowed and never crashes the scorer.

All three score spellF1 = 0.0000. That is the point of the metric: a constant guess has no
possessions in it, so no amount of luck on the frame counts can rescue it.

    python starter/make_dummy_submission.py --label A --name my_baseline
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402  (needs the path above)

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
DEFAULTS = (("dummy_dead", "D"), ("dummy_home", "H"), ("dummy_abstain", "X"))


def write_constant(name, label, n_frames, results_dir):
    """One row per video frame, all the same label. The whole submission format in four lines."""
    path = os.path.join(results_dir, "{}.csv".format(name))
    pd.DataFrame({
        "frame": np.arange(n_frames, dtype=np.int64),
        "label": np.full(n_frames, label, dtype="<U1"),
    }).to_csv(path, index=False)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", choices=("H", "A", "D", "X"),
                        help="write a single file of this label instead of the default three")
    parser.add_argument("--name", default="dummy",
                        help="file name to use with --label (without .csv)")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    args = parser.parse_args()

    n_frames = GAMES[GAME]["n_video_frames"]
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    wanted = [(args.name, args.label)] if args.label else list(DEFAULTS)
    for name, label in wanted:
        path = write_constant(name, label, n_frames, args.out)
        print("Wrote {} rows of '{}' to {}".format(n_frames, label, path))

    print("\nNow score them:  python evaluation/run_evaluation.py")
