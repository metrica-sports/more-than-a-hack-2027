"""Read the per-frame homography and move points between the pitch and the image.

The homography is what connects the two things you are given: it maps a position on the pitch to
the pixel that shows it, and back. That makes it the tool for looking at your own mistakes -- take
a spurious transition out of reports/<name>_transitions.csv, project the players onto that frame
of the video, and see what fooled the model.

    pip install -r requirements.txt          # numpy and pandas, nothing else
    python starter/load_homography.py        # summary, both sources

    data/tactical/BAR-ATH_tactical_homography.csv
    data/broadcast/BAR-ATH_broadcast_homography.csv

One row per video frame of that recording, `frame` counting from 0. For the tactical file that is
the same index as a submission row, so row i is the homography for the frame you label in row i.

    frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid

Columns 1..9 are the 3x3 matrix in row-major order, which is why load() can reshape them without
naming a single one.

WHICH WAY ROUND. The matrix maps NORMALISED PITCH COORDINATES (0-1) to image pixels. Those are the
same 0-1 coordinates the ATD reports, so an ATD position goes straight in with no conversion:

    x, y = project(H[frame], tracking_x, tracking_y)

Multiply by the pitch size in metres only if you want metres. Do not multiply before projecting.

VALID IS NOT DECORATION. Where `valid` is 0 the nine matrix columns are NaN, because no pitch
transform exists for that frame at all. The tactical recording is valid everywhere. The broadcast
recording is valid on 65.8% of frames -- the rest are replays, close-ups and crowd shots, in 122
stretches, the longest over a minute and a half. Check it before you use a matrix, or NaN will
check it for you.

`reprojection_error` is how far the fitted homography misses by, so LOWER IS BETTER and there is
no upper bound -- the broadcast file reaches 304. On the tactical file the median is 0.24 during
play and about 1.5 across halftime, when the camera is not pointed at the pitch. Use it as a soft
signal on top of `valid`: `valid` says whether a matrix exists, this says how much to trust it.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, DEFAULT_SOURCE, require, source_frames  # noqa: E402

# Params
MATRIX_COLUMNS = ["h{}{}".format(row, col) for row in range(3) for col in range(3)]


def load(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """(matrices, reprojection_error, valid) indexed by video frame.

    matrices is (n, 3, 3) float64, NaN wherever valid is False. The other two are (n,), the last
    boolean -- so `matrices[valid]` and `np.flatnonzero(valid)` both do what you would expect.
    """
    path = require(["homography"], game, data_dir, source)[0]
    table = pd.read_csv(path)

    missing = [name for name in ["frame", "reprojection_error", "valid"] + MATRIX_COLUMNS
               if name not in table.columns]
    if missing:
        raise ValueError("{}: missing columns {}".format(path, missing))

    matrices = table[MATRIX_COLUMNS].values.astype(np.float64).reshape(-1, 3, 3)
    return matrices, table["reprojection_error"].values, table["valid"].values.astype(bool)


def project(matrix, x, y):
    """Normalised pitch coordinates (0-1) -> image pixels. Scalars or arrays.

    NaN in, NaN out: an invalid frame's matrix is NaN, so forgetting to check `valid` produces
    obviously broken coordinates rather than plausible wrong ones.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    return ((matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator,
            (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator)


def unproject(matrix, pixel_x, pixel_y):
    """Image pixels -> normalised pitch coordinates (0-1). The inverse of project().

    Useful for the other direction of the debugging loop: click a point in a frame and ask where
    on the pitch it is.
    """
    return project(np.linalg.inv(matrix), pixel_x, pixel_y)


def pitch_corners(matrix):
    """The four pitch corners in pixels, clockwise from (0, 0). A one-line sanity check.

    If these do not form a sane quadrilateral around the visible pitch, the matrix is not what you
    think it is -- which is the fastest way to catch a convention mistake.
    """
    xs, ys = project(matrix, np.array([0.0, 1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0, 1.0]))
    return list(zip(xs.tolist(), ys.tolist()))


def check_alignment(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """Confirm the file really is one row per video frame, counting from 0.

    Cheap, and worth running against any new export rather than trusting the note above.
    """
    path = require(["homography"], game, data_dir, source)[0]
    frames = pd.read_csv(path, usecols=["frame"])["frame"].values
    expected = source_frames(source, game)
    return {
        "n_rows": len(frames),
        "n_video_frames": expected,
        "contiguous": bool(len(frames)) and np.array_equal(frames, np.arange(len(frames))),
        "agrees": len(frames) == expected and np.array_equal(frames, np.arange(len(frames))),
    }


if __name__ == '__main__':
    for source in ("tactical", "broadcast"):
        print("=" * 78)
        print(source)
        try:
            matrices, reprojection_error, valid = load(source=source)
        except IOError as error:
            print("  {}\n".format(error))
            continue

        report = check_alignment(source=source)
        print("  {} rows, one per video frame: {}".format(report["n_rows"], report["agrees"]))
        print("  valid   {:>7}  {:5.1f}%".format(int(valid.sum()), 100.0 * valid.mean()))
        print("  invalid {:>7}  {:5.1f}%   (matrix is NaN on these)".format(
            int((~valid).sum()), 100.0 * (~valid).mean()))
        if valid.any():
            usable = reprojection_error[valid]
            print("  reprojection error (lower is better): median {:.3f}   p99 {:.3f}".format(
                float(np.median(usable)), float(np.percentile(usable, 99))))

            frame = int(np.flatnonzero(valid)[0])
            centre_x, centre_y = project(matrices[frame], 0.5, 0.5)
            print("\n  first valid frame is {}".format(frame))
            print("    pitch centre  -> ({:8.1f}, {:8.1f}) px".format(centre_x, centre_y))
            for (x, y), corner in zip(pitch_corners(matrices[frame]),
                                      ("(0,0)", "(1,0)", "(1,1)", "(0,1)")):
                print("    corner {:<6} -> ({:8.1f}, {:8.1f}) px".format(corner, x, y))
        print()

    print("Coordinates go in normalised (0-1), the same as the ATD, and come out in pixels.")
    print("Check `valid` before using a matrix -- on the broadcast a third of frames have none.")
