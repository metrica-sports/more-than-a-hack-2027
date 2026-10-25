"""Every tunable number in one place.

Rule from RULES.md: none of these may be tuned on the public 5 minutes. Choose them from football
reasoning or from other games, and write the reason next to the value. The values below are
starting points, not results.
"""

# Ball gaps up to this many frames are bridged by linear interpolation (25 frames = 1 s).
MAX_BALL_GAP = 25

# Ball gaps longer than this are treated as "probably dead" in the v0 dead-ball prior.
DEAD_GAP = 75
DEAD_PROB_WHEN_LOST = 0.8

# Metres over which the home/away distance difference saturates into a confident vote.
DISTANCE_SCALE = 3.0

# Viterbi switch costs (log-probability units). Larger = fewer, longer possessions.
# spellF1 punishes flicker far more than macroF1 does, so these matter more than any feature.
SWITCH_HOME_AWAY = 12.0
SWITCH_DEAD = 8.0

# Kickoff of the 2nd half; frames from the start of the graded block to here are all `D`
# (FAQ.md). Source: starter/load_smart_tagging.py KICKOFF_VIDEO_FRAMES.
SECOND_HALF_KICKOFF = 113742
