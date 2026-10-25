"""Probability that the ball is dead, per frame. v0 is a placeholder -- this is where to work.

Ideas, roughly by expected value:
  * Smart Tagging SET PIECES intervals (restarts) as a prior; throw-ins have no team, see DATA.md.
  * Ball outside the pitch rectangle, ball speed ~0 while players walk to the spot.
  * Player layout: set-piece shapes (wall, corner cluster) from the tracking.
"""

import numpy as np

from games import half_video_blocks
from . import config


def dead_probability(n_frames, ball_confidence, game="BAR-ATH"):
    """(N,) float in [0, 1]."""
    p = np.full(n_frames, 0.05, dtype=np.float32)

    # v0: a long stretch with no ball is "probably dead". Crude; the baseline does the same.
    lost = ball_confidence == 0
    run_start = None
    for i in range(n_frames + 1):
        if i < n_frames and lost[i]:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            if i - run_start > config.DEAD_GAP:
                p[run_start:i] = config.DEAD_PROB_WHEN_LOST
            run_start = None

    # The 2nd half is graded from the first frame of the block but kicks off later: all `D`.
    for half, first, _ in half_video_blocks(game):
        if half == 2:
            p[first:config.SECOND_HALF_KICKOFF] = 1.0
    return p
