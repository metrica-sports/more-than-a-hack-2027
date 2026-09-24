"""Read the Smart Tagging XML and convert its intervals to video frames.

Smart Tagging is Metrica's automatic tactical annotation of the tactical video: phases of play and
set pieces, each an interval with a code and a set of labels. It is a SportsCode XML file, so this
needs nothing but the standard library.

    python starter/load_smart_tagging.py

What it is good for: every phase carries a Team tag, and every set piece is a restart at the end
of a dead-ball stretch. That is a free prior on both halves of the label you are producing.

What it is not: ground truth. It is inferred automatically and contains false positives and false
negatives, and its intervals are *phases*, not possessions -- a Build Up can span one possession,
stop short of one, or cover two. Copy an interval across as a label and you inherit its errors.

TIME BASE, verified on the delivered file: <start> and <end> are seconds from the first frame of
the tactical video, so seconds x 25 is a video frame and nothing else is needed. Both tagged
kickoffs contain the real one. check_time_base() is that check, kept so it runs again if the
export ever changes.
"""

import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402

# Params
# Codes in the file are upper case ("SET PIECES"), tag values are title case ("Kick Off (Start)").
# Matched case-insensitively here so a change in the export does not silently stop matching.
KICKOFF_CODE = "set pieces"
KICKOFF_TYPE = "kick off (start)"

# Where each half actually starts. Both numbers are published, in DATA.md and FAQ.md: the first is
# the first graded frame of the match, the second is the second-half kickoff. The tagging marks a ~5 s window around the restart rather than the instant, so the
# check below is containment, not equality.
KICKOFF_VIDEO_FRAMES = {"1st Half": 11185, "2nd Half": 113742}


def pattern_path(game=GAME, data_dir=None):
    """Path to the Smart Tagging XML, with a readable error if it is not there."""
    return require(["smart_tagging"], game, data_dir)[0]


def load(game=GAME, data_dir=None, fps=None):
    """[{code, start_frame, end_frame, start_s, end_s, tags}] in file order.

    Frames are half-open on the right the way the rest of this repository treats intervals, so
    end_frame is the first frame *after* the phase. Rounding is to the nearest frame: the tags are
    accurate to about a second, so a half-frame either way is not the error that matters.
    """
    fps = fps or GAMES[game]["fps"]
    out = []
    for instance in ET.parse(pattern_path(game, data_dir)).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        out.append({
            "code": instance.findtext("code"),
            "start_s": start_s,
            "end_s": end_s,
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            # A group can legitimately appear more than once; last one wins, which is fine for
            # the single-valued groups (Team, Half, Type) this file uses.
            "tags": {label.findtext("group"): label.findtext("text")
                     for label in instance.findall("label")},
        })
    return out


def check_time_base(events):
    """Do the tagged kickoffs land on the real ones? One row per half.

    Containment rather than equality: the tagging marks a window around the restart, not the
    instant of it, so "the real kickoff frame falls inside the tagged interval" is the strongest
    claim the data supports -- and it is enough to prove the two clocks agree.
    """
    kickoffs = [event for event in events
                if (event["code"] or "").strip().lower() == KICKOFF_CODE
                and (event["tags"].get("Type") or "").strip().lower() == KICKOFF_TYPE]

    rows = []
    for half in sorted(KICKOFF_VIDEO_FRAMES):
        truth = KICKOFF_VIDEO_FRAMES[half]
        tagged = [event for event in kickoffs if event["tags"].get("Half") == half]
        hit = next((event for event in tagged
                    if event["start_frame"] <= truth <= event["end_frame"]), None)
        rows.append({
            "half": half,
            "true_kickoff_frame": truth,
            "tagged": [(event["start_frame"], event["end_frame"]) for event in tagged],
            "contains_kickoff": hit is not None,
        })
    return {
        "n_kickoff_tags": len(kickoffs),
        "halves": rows,
        "agrees": all(row["contains_kickoff"] for row in rows),
    }


if __name__ == '__main__':
    events = load()
    print("loaded {} Smart Tagging instances".format(len(events)))

    counts = {}
    for event in events:
        counts[event["code"]] = counts.get(event["code"], 0) + 1
    print("\ncodes")
    for code in sorted(counts, key=lambda c: -counts[c]):
        print("  {:<34} {:>4}".format(code, counts[code]))

    teams = sorted({event["tags"].get("Team") for event in events} - {None})
    print("\nTeam tag values: {}".format(teams))
    print("  map these to H and A yourself -- the XML names clubs, the submission wants a side")

    report = check_time_base(events)
    print("\ntime base check  ({} 'Kick Off (Start)' tags in the file)".format(
        report["n_kickoff_tags"]))
    for row in report["halves"]:
        print("  {:<10} real kickoff frame {:>6}   tagged {}   contains it: {}".format(
            row["half"], row["true_kickoff_frame"], row["tagged"], row["contains_kickoff"]))
    if report["agrees"]:
        print("\n  OK -- the XML clock is zero at the first frame of the tactical video.")
        print("  Seconds x {} gives you a video frame directly.".format(GAMES[GAME]["fps"]))
    else:
        print("\n  MISMATCH -- the XML is on a different clock than the video frame index.")
        print("  Do not add a constant to make it fit. Report it, so every team gets the")
        print("  same time base rather than each inventing their own.")
