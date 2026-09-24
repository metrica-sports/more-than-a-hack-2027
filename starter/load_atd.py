"""Load the provided Automatic Tracking Data and line it up with the video frames you submit.

The ATD is Metrica's FIFA/EPTS export -- a metadata XML plus a raw text file:

    data/tactical/BAR-ATH_tactical_atd.xml
    data/tactical/BAR-ATH_tactical_atd.txt

games.asset_paths() builds both, so nothing here spells a filename out.

The match was recorded twice and each recording has its own ATD. This module defaults to the
tactical one, because that is the recording a submission is indexed to. The broadcast ATD is a
separate 164488-frame dataset on its own timeline:

    load(source="broadcast")

kloppy reads that pair directly, so this module is thin on purpose. What it adds is the part that
is easy to get wrong and expensive to notice: the join between tracking frames and video frames.

    pip install -r requirements-starter.txt
    python starter/load_atd.py                  # prints a summary and a few aligned frames

What you actually need to know
------------------------------
The ATD is **already on the video's timeline**, one record per video frame, 199319 of them. The
only catch is that it counts from 1 where your submission counts from 0:

    video_frame = frame_id - 1

That is the whole alignment story for participants. You do not need the per-half offsets in
games.py -- those describe the separate counter the ground truth was derived from, which you never
see. Verified on the delivered file: 199319 records numbered 1..199319, with 21-22 tracked objects
during play and almost none across the halftime break, exactly where the ground truth says the
match is not being played.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DEFAULT_SOURCE, require, source_frames  # noqa: E402

# Params
# Raw ATD line: "<frame_id>:<player>;<player>;...:<ball>". Only the part before the first
# separator is needed to check the alignment.
FIELD_SEP = ":"


def atd_paths(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """(meta_data, raw_data) paths, or a readable error naming what is missing."""
    return tuple(require(["atd_meta", "atd_raw"], game, data_dir, source))


def load(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """The ATD as a kloppy TrackingDataset.

    Coordinates come back in kloppy's Metrica convention. The real pitch is
    GAMES[game]["pitch"] metres, which is what you want for anything measured in metres rather
    than in fractions of a pitch.
    """
    try:
        from kloppy import metrica
    except ImportError:
        raise ImportError(
            "kloppy is not installed. It is optional -- the evaluation needs only numpy and "
            "pandas -- but this loader needs it:\n    pip install -r requirements-starter.txt")

    meta, raw = atd_paths(game, data_dir, source)
    return metrica.load_tracking_epts(meta_data=meta, raw_data=raw)


def video_frame_of(frame_id):
    """Video frame -- and therefore submission row -- of an ATD record."""
    return int(frame_id) - 1


def frame_id_of(video_frame):
    """ATD frame_id of a video frame. The inverse of video_frame_of."""
    return int(video_frame) + 1


def check_alignment_raw(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """The same check as check_alignment, straight off the raw file. No kloppy, no full load.

    Loading 199319 frames x 46 slots through kloppy to count them costs minutes and gigabytes;
    the raw file answers the same question in one streaming pass, because every line begins with
    its own frame id. Use this one unless you already have a dataset in hand.
    """
    raw_path = require(["atd_raw"], game, data_dir, source)[0]
    expected = source_frames(source, game)

    first = last = None
    count = 0
    with open(raw_path) as handle:
        for line in handle:
            frame_id = int(line[:line.index(FIELD_SEP)])
            if first is None:
                first = frame_id
            last = frame_id
            count += 1

    contiguous = count > 0 and last - first + 1 == count
    return {
        "n_records": count,
        "n_video_frames": expected,
        "frame_id_range": (first, last),
        "contiguous": contiguous,
        "first_video_frame": video_frame_of(first) if first is not None else None,
        "agrees": count == expected and contiguous and first == 1,
    }


def check_alignment(dataset, game=GAME, source=DEFAULT_SOURCE):
    """Confirm the ATD really is one record per video frame, numbered from 1.

    Cheap, and worth running once against any new export rather than trusting the note above.
    Two claims, both falsifiable: the record count equals the video length, and the ids are
    exactly 1..n with nothing missing.
    """
    expected = source_frames(source, game)
    ids = sorted(int(record.frame_id) for record in dataset.records)
    contiguous = bool(ids) and ids == list(range(ids[0], ids[0] + len(ids)))
    return {
        "n_records": len(ids),
        "n_video_frames": expected,
        "frame_id_range": (ids[0], ids[-1]) if ids else (None, None),
        "contiguous": contiguous,
        "first_video_frame": video_frame_of(ids[0]) if ids else None,
        "agrees": len(ids) == expected and contiguous and bool(ids) and ids[0] == 1,
    }


if __name__ == '__main__':
    dataset = load()
    print("loaded {} tracking records".format(len(dataset.records)))
    print("teams: {}  ->  H = {}, A = {}".format(
        [str(team) for team in dataset.metadata.teams],
        dataset.metadata.teams[0], dataset.metadata.teams[1]))
    print("pitch: {} x {} m from the ATD metadata, {} x {} m in games.py".format(
        dataset.metadata.pitch_dimensions.pitch_length,
        dataset.metadata.pitch_dimensions.pitch_width,
        *GAMES[GAME]["pitch"]))
    print("video: {} frames at {} fps".format(
        GAMES[GAME]["n_video_frames"], GAMES[GAME]["fps"]))

    report = check_alignment(dataset)
    print("\nalignment check")
    print("  records {}, video frames {}".format(report["n_records"], report["n_video_frames"]))
    print("  frame_id range {}..{}, contiguous: {}".format(
        report["frame_id_range"][0], report["frame_id_range"][1], report["contiguous"]))
    if report["agrees"]:
        print("\n  OK -- one record per video frame, numbered from 1.")
        print("  video_frame = frame_id - 1. Nothing else to do.")
    else:
        print("\n  MISMATCH -- this export is not one record per video frame from 1.")
        print("  Do not invent a correction. Report it, so every team gets the same alignment.")

    print("\nsome landmarks, as video frames:")
    for video_frame, what in ((11185, "1st half kickoff, first graded frame"),
                              (18684, "end of the public 5-minute window"),
                              (100000, "halftime -- no play"),
                              (113742, "2nd half kickoff"),
                              (190197, "last graded frame")):
        print("  video {:>6}  = ATD frame_id {:>6}   {}".format(
            video_frame, frame_id_of(video_frame), what))
