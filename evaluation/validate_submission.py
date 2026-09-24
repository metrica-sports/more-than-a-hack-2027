"""Check a submission file before you send it. No ground truth needed.

Run this on every file you are about to submit. It applies exactly the same validation the
official scorer applies, so a file that passes here will not be rejected there -- and a file that
fails here would have been silently dropped from the board with a one-line reason nobody would
have read in time.

    python evaluation/validate_submission.py my_submission.csv
    python evaluation/validate_submission.py my_submission.csv --rows 199319

On success it also prints what the file actually says -- label counts, how many possessions it
reports and how long they are. That second half is worth reading. A file can be perfectly valid
and still claim the match had 900 possessions of 5 seconds each, and the format checker is the
earliest place you can notice.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import possession_metrics as pm  # noqa: E402  (needs the path above)
from games import GAME, GAMES  # noqa: E402


def describe(codes, fps):
    """What the submission claims, in the terms a football analyst would use.

    Runs are the file's own non-X spans, because without a ground truth there is nothing else to
    take them from. That makes an abstention a gap, so 'H X X H' counts as two possessions here
    where the official scorer, which knows the truth is continuous there, fills the gap and counts
    one. The difference only shows up in files that actually contain X, and it is flagged below.
    """
    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    n_alive = int(np.sum((codes == pm.H) | (codes == pm.A)))
    mean_duration = (n_alive / float(fps) / len(spells)) if len(spells) else 0.0
    return len(spells), mean_duration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission", help="the frame,label CSV to check")
    parser.add_argument("--rows", type=int, default=GAMES[GAME]["n_video_frames"],
                        help="expected row count (default: this match's video length)")
    args = parser.parse_args()

    if not os.path.isfile(args.submission):
        print("FAIL  no such file: {}".format(args.submission))
        sys.exit(1)

    try:
        table = pm.read_label_csv(args.submission)
    except Exception as error:  # unparseable CSV, missing columns, unreadable file
        print("FAIL  {}".format(os.path.basename(args.submission)))
        print("      could not read: {}".format(error))
        print("\nExpected a CSV with a header line 'frame,label' and one row per video frame.")
        sys.exit(1)

    reason = pm.validate_submission(table, args.rows)
    if reason:
        print("FAIL  {}".format(os.path.basename(args.submission)))
        print("      {}".format(reason))
        print("\nSee SUBMISSION.md for the contract. The usual causes are an off-by-one at the "
              "start or end of\nthe file, a missing header, and blank cells where the system had "
              "no answer -- write X instead.")
        sys.exit(1)

    codes = pm.label_codes(table)
    fps = GAMES[GAME]["fps"]
    n_spells, mean_duration = describe(codes, fps)

    print("OK    {}".format(os.path.basename(args.submission)))
    print("      {} rows, frame 0..{}".format(len(codes), len(codes) - 1))
    print("\n      label counts")
    for index, name in enumerate(pm.LABELS):
        count = int(np.sum(codes == index))
        print("        {}  {:>7}  {:5.1f}%".format(name, count, 100.0 * count / len(codes)))
    print("\n      reports {} possessions of mean {:.1f} s".format(n_spells, mean_duration))
    if int(np.sum(codes == pm.X)):
        print("      (counted with each X as a gap; the scorer bridges abstentions inside a real")
        print("       possession, so its count can be lower than this one)")
    print("\nValid. Note this checks the format only -- it says nothing about whether the labels "
          "are right.\nScore it against the public ground truth with run_evaluation.py.")
