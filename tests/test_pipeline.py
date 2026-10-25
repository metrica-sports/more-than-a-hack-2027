"""Smoke tests on synthetic data -- they need no match data:  python -m unittest discover tests"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession.features import interpolate_ball, team_distances  # noqa: E402
from possession.pipeline import predict  # noqa: E402
from possession.states import A, H, viterbi  # noqa: E402
from games import GAMES, GAME, half_video_blocks  # noqa: E402


class StatesTest(unittest.TestCase):
    def test_viterbi_removes_flicker(self):
        em = np.log(np.array([[.9, .05, .05]] * 40 + [[.05, .9, .05]] * 40))
        em[20] = np.log([.05, .9, .05])               # one-frame glitch
        path = viterbi(em, 5.0, 5.0)
        self.assertTrue((path[:40] == H).all() and (path[40:] == A).all())


class FeaturesTest(unittest.TestCase):
    def test_interpolation_respects_max_gap(self):
        ball = np.full((10, 2), np.nan, dtype=np.float32)
        ball[0], ball[3], ball[9] = (0, 0), (3, 0), (9, 0)
        filled, conf = interpolate_ball(ball, max_gap=3)
        self.assertAlmostEqual(float(filled[1, 0]), 1.0, places=5)
        self.assertTrue(np.isnan(filled[5, 0]))        # gap of 5 > 3
        self.assertEqual(float(conf[5]), 0.0)

    def test_team_distances(self):
        ball = np.array([[0, 0]], dtype=np.float32)
        players = np.array([[[1, 0], [5, 0], [np.nan, np.nan]]], dtype=np.float32)
        home, away = team_distances(ball, players, np.array([0, 1, 1]))
        self.assertEqual((float(home[0]), float(away[0])), (1.0, 5.0))


class PipelineTest(unittest.TestCase):
    def test_labels_cover_every_frame_and_respect_graded_blocks(self):
        n = GAMES[GAME]["n_video_frames"]
        ball = np.full((n, 2), 50.0, dtype=np.float32)
        players = np.full((n, 2, 2), np.nan, dtype=np.float32)
        players[:, 0] = (49, 50)                      # home slot next to the ball
        players[:, 1] = (60, 50)
        labels = predict(ball, players, np.array([0, 1]))
        self.assertEqual(len(labels), n)
        (_, f1, l1), (_, f2, l2) = half_video_blocks(GAME)
        self.assertEqual(labels[0], "X")
        self.assertEqual(labels[l1 + 1], "X")
        self.assertEqual(labels[f1 + 100], "H")
        self.assertEqual(labels[f2 + 10], "D")        # 2nd half pre-kickoff
        self.assertEqual(labels[l2 + 1], "X")


if __name__ == "__main__":
    unittest.main()
