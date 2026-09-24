"""Score every submission in ./results/ against the ground truth and print a ranked scoreboard.

Submissions are frame,label CSVs with one row per video frame, exactly as long as the ground
truth, using the H/A/D/X alphabet. Anything in ./results/ ending in .csv is treated as an entry.

Ranking is on spellF1 -- the mean possession-spell F1 over IoU 0.3..0.7 -- ties broken by macro-F1
over {H, A, D}, then transition F1 at the loose tolerance, then mean absolute latency, then
filename so the order is never arbitrary. See possession_metrics.py for why possessions are scored
as intervals rather than frames, why macro-F1 and not accuracy, and for the noise floor on
comparisons.

A submission that fails validation is skipped and listed at the bottom with the reason, rather
than aborting the run -- one malformed file must not hide everyone else's results. The exit code
is 1 if anything was skipped. A broken *ground truth* is fatal, because it invalidates every
score rather than one.

The ground truth is chosen by name, not by path: --gt public is the 5-minute slice that ships
with the repository and is the default, --gt private is the full match and only exists on an
organiser's machine. A path still works if you have one somewhere else.

Usage:
    python run_evaluation.py                         # scored on the public 5 minutes
    python run_evaluation.py --gt private            # organisers: the full match
    python run_evaluation.py --bootstrap 1000        # confidence intervals; do this before
                                                     # believing any gap on the 5-minute sample
    python run_evaluation.py --results some/dir --gt some/other_gt.csv
"""

import argparse
import glob
import json
import os
import sys

import possession_metrics as pm

# Load Data
HERE = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(HERE, "gt")
RESULTS_DIR = os.path.join(HERE, "results")

# Params
# --gt takes a name, not a path. 'public' is the 5-minute slice that ships with the repository;
# 'private' is the full match and exists only on an organiser's machine. Naming them rather than
# globbing for whatever is in gt/ means the two can sit side by side without the default quietly
# changing meaning when the second one appears.
GT_ALIASES = {
    "public": ("*_possession_gt_public.csv",
               "The public ground truth ships with this repository. If it is missing, re-clone."),
    "private": ("*_possession_gt.csv",
                "The full-match ground truth is private and is not in this repository. If you are "
                "an organiser, put it in gt/; otherwise you want --gt public."),
}
DEFAULT_GT = "public"

OUT_DIR = HERE
REPORTS_DIRNAME = "reports"
NAME_WIDTH = 24  # longer submission names are truncated with a trailing ~ to keep the table narrow

# Which metric orders the board. spell_f1_mean, because possession is an interval concept in
# football and that metric counts each possession once, where macro_f1 is dominated by the great
# majority of frames sitting in the middle of a possession. Set to "macro_f1" to rank on the
# frame-level view instead; everything else on the board is unaffected either way.
#
# The switch costs nothing in discrimination: spell_f1_mean's bootstrap band is 3-4x wider than
# macro_f1's (~0.036 vs ~0.01) but the gaps widen in proportion, so 5 of 10 adjacent pairs are
# statistically indistinguishable under either choice.
RANK_METRIC = "spell_f1_mean"
# Noise band of the ranking metric, printed as a caveat so a third-decimal gap is not read as a
# result. Keyed by metric so it stays correct if RANK_METRIC changes.
RANK_NOISE_BAND = {"spell_f1_mean": pm.SPELL_NOISE_BAND, "macro_f1": 0.01}

# SPELL_NOISE_BAND was measured over a full match's possessions. The band scales with how many
# there are, so it is not transferable to a short ground truth: on the 5-minute slice shipped with
# this repository the real band is several times wider, and printing "~0.04" would understate it
# by an order of magnitude.
#
# Below this many possessions the header says the quoted band does not apply rather than quoting a
# number that would mislead, and switches to the measured spread as soon as --bootstrap provides
# one. The threshold is a round number chosen well above the public slice and well below a full
# match; nothing depends on its exact value.
SPELL_NOISE_BAND_MIN_POSSESSIONS = 100

# Flat columns for scoreboard.csv. Tau-dependent ones are appended per tolerance in
# scoreboard_columns(), so changing TAUS does not mean editing this tuple.
# The four columns the printed board leads with come first and use the board's own header names,
# so a column found on screen is findable in the CSV by the same word. The rest follow in rough
# order of how often they get looked at.
BASE_COLS = (
    "rank", "submission", "status", "reason",
    "macro_f1", "spell_f1_mean", "poss", "dur",
    "n_gt_spells", "gt_mean_spell_duration_s", "spell_count_error",
    "spell_f1_at_headline", "spell_iou_headline",
    "spell_bootstrap_low", "spell_bootstrap_high",
    "bootstrap_low", "bootstrap_high", "kappa", "accuracy",
    "f1_H", "f1_A", "f1_D",
    "precision_H", "recall_H", "precision_A", "recall_A", "precision_D", "recall_D",
    "alive_dead_f1", "team_acc_gt_alive", "team_acc_both_alive",
    "gt_home_share", "pred_home_share", "share_err", "share_defined",
    "mean_latency", "median_latency", "mean_abs_latency", "p90_abs_latency",
    "n_matched", "n_missed", "n_spurious", "n_ambiguous",
    "n_gt_segments", "n_pred_segments", "fragmentation",
    "n_graded", "n_abstain_on_graded",
)
TRANSITION_COLS = ("trans_f1", "trans_precision", "trans_recall")


def scoreboard_columns(taus):
    return BASE_COLS + tuple("{}_{}".format(col, tol) for tol in taus for col in TRANSITION_COLS)


def find_gt(gt_arg):
    """Resolve --gt: the word 'public', the word 'private', or a path.

    Two ground truths live in gt/ and which one you meant is never obvious from a filename, so
    they are named rather than pointed at. The default is public, because that is the only one a
    participant has and the only one that is safe to leave as a default.
    """
    if gt_arg not in GT_ALIASES:
        if not os.path.isfile(gt_arg):
            raise IOError(
                "ground truth not found: {}\n--gt takes 'public', 'private', or a path to a "
                "frame,label CSV.".format(gt_arg))
        return gt_arg

    pattern, description = GT_ALIASES[gt_arg]
    # '*_possession_gt.csv' does not match '*_possession_gt_public.csv', so the two patterns are
    # mutually exclusive and neither picks up the ungraded '_reference.csv' companion.
    candidates = sorted(glob.glob(os.path.join(GT_DIR, pattern)))
    if not candidates:
        raise IOError("no {} ground truth in {} (looked for {}).\n{}".format(
            gt_arg, GT_DIR, pattern, description))
    if len(candidates) > 1:
        raise IOError("several {} ground truths in {}; pass an explicit path:\n  {}".format(
            gt_arg, GT_DIR, "\n  ".join(os.path.basename(c) for c in candidates)))
    return candidates[0]


def load_meta(gt_path):
    """The sidecar written beside the ground truth, or {} if it is not there.

    One rule for both naming schemes: <name>.csv -> <name>_meta.json.
    """
    meta_path = gt_path[:-len(".csv")] + "_meta.json" if gt_path.endswith(".csv") else ""
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path) as handle:
        return json.load(handle)


def score_submission(path, gt_codes, taus, bootstrap_n):
    """(result, reason). Exactly one of the two is None."""
    try:
        table = pm.read_label_csv(path)
    except Exception as error:  # unparseable CSV, missing columns, unreadable file
        return None, "could not read: {}".format(error)

    reason = pm.validate_submission(table, len(gt_codes))
    if reason:
        return None, reason
    return pm.evaluate(gt_codes, pm.label_codes(table), taus=taus,
                       bootstrap_n=bootstrap_n), None


def rank_key(entry, loose_tau):
    """RANK_METRIC desc, then macro-F1, then transition F1, then latency asc, name asc.

    macro-F1 is the first tie-break rather than the name because two submissions can tie exactly
    on the ranking metric: team_swapped and always_home both score 0.0000 on spell_f1_mean, and
    macro-F1 separates them meaningfully (0.3333 against 0.1627) where alphabetical order would
    be arbitrary.
    """
    result = entry["result"]
    latency = result["transitions"][loose_tau]["mean_abs_latency"]
    return (-result[RANK_METRIC],
            -result["macro_f1"],
            -result["transitions"][loose_tau]["f1"],
            latency if latency is not None else float("inf"),
            entry["name"])


def print_scoreboard(entries, skipped, gt_path, meta, loose_tau):
    """The ranked table. 93 columns wide, so it survives a 100-column terminal.

    macroF1 and spellF1 sit next to each other on purpose: they are the two headline numbers and
    answer different questions -- "right label at this frame" versus "right possessions" -- so
    they are meant to be read as a pair. kappa and acc were dropped from the display to make room
    and remain in scoreboard.csv and the detail block; neither was ever rankable.
    """
    n_segments = meta.get("n_segments")
    reference = entries[0]["result"] if entries else None
    print("scoreboard  {}  {} frames  {} graded{}".format(
        os.path.relpath(gt_path, HERE),
        meta.get("n_video_frames", "?"), meta.get("n_graded_frames", "?"),
        "  {} gt segments".format(n_segments) if n_segments else ""))
    if reference is not None:
        # The poss and dur columns are meaningless without the truth to compare them against.
        print("ground truth has {} possessions of mean {:.1f} s  <- compare the poss and dur "
              "columns".format(reference["n_gt_spells"], reference["gt_mean_spell_duration_s"]))
    # Without this the reader will treat a third-decimal difference as a result. The band belongs
    # to whichever metric is ranking, so it is looked up rather than hard-coded.
    band = RANK_NOISE_BAND.get(RANK_METRIC)
    measured = [entry["result"]["spell_bootstrap_high"] - entry["result"]["spell_bootstrap_low"]
                for entry in entries
                if entry["result"].get("spell_bootstrap_low") is not None]
    if reference is None or band is None:
        pass
    elif RANK_METRIC == "spell_f1_mean" and measured:
        # A measured band beats a quoted one. The widest interval on the board is the honest
        # figure: a gap has to clear the noisiest submission it separates, not the quietest.
        print("ranked on {}; gaps below ~{:.2f} are within noise (widest bootstrap interval, "
              "{} possessions, {} draws)".format(
                  RANK_METRIC, max(measured), reference["n_gt_spells"],
                  entries[0]["result"]["bootstrap_n"]))
    elif (RANK_METRIC == "spell_f1_mean"
            and reference["n_gt_spells"] < SPELL_NOISE_BAND_MIN_POSSESSIONS):
        print("ranked on {}; the ~{:.2f} noise band was calibrated over a full match and this "
              "ground truth has only {} possessions -- run --bootstrap 1000 for the real band"
              .format(RANK_METRIC, band, reference["n_gt_spells"]))
    else:
        print("ranked on {}; gaps below ~{:.2f} are within noise ({} possessions, {} segments)"
              .format(RANK_METRIC, band, reference["n_gt_spells"], n_segments))
    print("spellF1 = mean possession-spell F1 over IoU {}; transF1/lat_f at +/-{} frames".format(
        "/".join("{:.1f}".format(i) for i in pm.SPELL_IOUS), loose_tau))

    header = ("{:>4}  {:<{w}}  {:>7}  {:>7}  {:>5}  {:>6}  {:>7}  {:>5}  {:>6}  {:>4}".format(
        "rank", "submission", "macroF1", "spellF1", "poss", "dur", "transF1", "lat_f",
        "shareE", "frag", w=NAME_WIDTH))
    print("\n" + header)
    print("{}  {}  {}  {}  {}  {}  {}  {}  {}  {}".format(
        "-" * 4, "-" * NAME_WIDTH, "-" * 7, "-" * 7, "-" * 5, "-" * 6, "-" * 7, "-" * 5,
        "-" * 6, "-" * 4))

    for rank, entry in enumerate(entries, start=1):
        result = entry["result"]
        transition = result["transitions"][loose_tau]
        print("{:>4}  {:<{w}}  {:>7.4f}  {:>7.4f}  {:>5}  {:>6.1f}  {:>7.4f}  {:>5}  {:>6}  "
              "{:>4}".format(
                  rank, truncate(entry["name"]), result["macro_f1"], result["spell_f1_mean"],
                  result["n_pred_spells"], result["mean_spell_duration_s"], transition["f1"],
                  signed(transition["mean_latency"]),
                  "{:.3f}".format(result["share_err"]) if result["share_defined"] else "  --",
                  "{:.2f}".format(result["fragmentation"]), w=NAME_WIDTH))

    for entry in skipped:
        print("{:>4}  {:<{w}}  {:>7}  {:>7}  {:>5}  {:>6}  {:>7}  {:>5}  {:>6}  {:>4}".format(
            "-", truncate(entry["name"]), "--", "--", "--", "--", "--", "--", "--", "--",
            w=NAME_WIDTH))
        print("        SKIPPED: {}".format(entry["reason"]))

    print_headline_agreement(entries)


def print_headline_agreement(entries):
    """Say whether the other headline metric would order the field differently.

    This is the reason both are reported. Agreement means the ranking does not depend on which
    view you take. Disagreement means a submission is trading per-frame label accuracy against
    possession structure -- typically one that is punctual but fragments possessions against one
    that is slightly late but keeps them whole -- and that tradeoff is a judgement call no single
    number can make. Movement is reported relative to the board, i.e. from RANK_METRIC's order
    to the other metric's.
    """
    if len(entries) < 2:
        return

    other = "macro_f1" if RANK_METRIC == "spell_f1_mean" else "spell_f1_mean"
    label = {"macro_f1": "macroF1", "spell_f1_mean": "spellF1"}

    # Competition ranks, so equal scores share a rank. Without this two submissions tied at the
    # same value look like they swapped places, purely because the sort broke the tie by name --
    # which is exactly what happens to team_swapped and always_home, both at spellF1 0.0000.
    board_rank = competition_ranks([e["result"][RANK_METRIC] for e in entries])
    other_rank = competition_ranks([e["result"][other] for e in entries])

    # Printing both full orderings needs ~200 characters once there are a dozen submissions, and
    # the reader then has to diff two lists by eye. Showing only what moved, worst first, is the
    # same information in a form that fits and points straight at what to look at.
    moves = [(abs(other_rank[i] - board_rank[i]), entries[i]["name"], board_rank[i],
              other_rank[i]) for i in range(len(entries)) if other_rank[i] != board_rank[i]]
    if not moves:
        print("\n{} and {} agree on the ordering, so the ranking does not hinge on which view "
              "you take.".format(label[RANK_METRIC], label[other]))
        return
    moves.sort(key=lambda m: (-m[0], m[2]))

    print("\n{} would order the field differently -- these move, biggest first:".format(
        label[other]))
    for delta, name, on_board, by_other in moves:
        print("  {:<26} board #{:<3} -> {} #{:<3}  {} {}".format(
            truncate(name), on_board, label[other], by_other,
            "up" if by_other < on_board else "DOWN", delta))
    print("  one that rises under {} is trading possession structure for per-frame accuracy"
          .format(label[other]) if RANK_METRIC == "spell_f1_mean" else
          "  one that rises under {} keeps possessions whole at some per-frame cost"
          .format(label[other]))


def competition_ranks(values):
    """1-based ranks, highest value first, with ties sharing a rank (1, 2, 2, 4)."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0] * len(values)
    position = 0
    while position < len(order):
        last = position
        while last + 1 < len(order) and values[order[last + 1]] == values[order[position]]:
            last += 1
        for index in range(position, last + 1):
            ranks[order[index]] = position + 1
        position = last + 1
    return ranks


def print_detail(entry, taus):
    """Everything the ranked table has no room for, one block per submission."""
    result = entry["result"]
    print("\n{}".format(entry["name"]))
    print("  macro F1 {:.4f}  ({})   kappa {:.4f}  acc {:.4f}".format(
        result["macro_f1"],
        "  ".join("{} {:.4f}".format(name, result["per_class"][name]["f1"])
                  for name in pm.SCORED),
        result["kappa"], result["accuracy"]))
    if result["bootstrap_low"] is not None:
        print("  bootstrap macroF1 {}-{} pct  {:.4f} .. {:.4f}   ({} segments, {} draws)".format(
            pm.BOOTSTRAP_PERCENTILES[0], pm.BOOTSTRAP_PERCENTILES[1],
            result["bootstrap_low"], result["bootstrap_high"],
            result["n_gt_segments"], result["bootstrap_n"]))

    print("  per class          precision   recall       F1   support")
    for name in pm.SCORED:
        stats = result["per_class"][name]
        print("    {}              {:>9.4f} {:>8.4f} {:>8.4f} {:>9}".format(
            name, stats["precision"], stats["recall"], stats["f1"], stats["support"]))

    print("  confusion  rows=GT cols=pred  {}".format(
        "".join("{:>8}".format(name) for name in pm.LABELS)))
    for row, name in enumerate(pm.LABELS):
        print("                          {}    {}".format(
            name, "".join("{:>8}".format(v) for v in result["confusion"][row])))

    print("  transitions  gt {}  pred {}".format(
        result["transitions"][taus[0]]["n_gt"], result["transitions"][taus[0]]["n_pred"]))
    for tol in taus:
        transition = result["transitions"][tol]
        print("      tau={:<3} P {:.4f}  R {:.4f}  F1 {:.4f}   matched {}  missed {}  "
              "spurious {}".format(tol, transition["precision"], transition["recall"],
                                   transition["f1"], transition["n_matched"],
                                   transition["n_missed"], transition["n_spurious"]))
    loose = result["transitions"][taus[-1]]
    print("  latency      mean {}  median {}  p90|lat| {}  ambiguous {}".format(
        signed(loose["mean_latency"]), signed(loose["median_latency"]),
        "--" if loose["p90_abs_latency"] is None else "{:.0f}".format(loose["p90_abs_latency"]),
        loose["n_ambiguous"]))
    if loose["n_ambiguous"]:
        # Kept on its own line so the figures above stay inside 100 columns.
        print("               windows overlap, so treat the mean as indicative")

    print("  possessions  gt {}  pred {}  ({:+d})   mean duration {:.1f} s vs {:.1f} s "
          "({:+.0f}%)".format(
              result["n_gt_spells"], result["n_pred_spells"], result["spell_count_error"],
              result["mean_spell_duration_s"], result["gt_mean_spell_duration_s"],
              100.0 * (result["mean_spell_duration_s"] - result["gt_mean_spell_duration_s"])
              / result["gt_mean_spell_duration_s"]
              if result["gt_mean_spell_duration_s"] else 0.0))
    print("  spell F1     mean {:.4f}   {}".format(
        result["spell_f1_mean"],
        "  ".join("@{:.1f} {:.4f}".format(iou, result["spell_f1"][iou]["f1"])
                  for iou in pm.SPELL_IOUS)))
    if result["spell_bootstrap_low"] is not None:
        print("      bootstrap {}-{} pct  {:.4f} .. {:.4f}   ({} possessions, {} draws)".format(
            pm.BOOTSTRAP_PERCENTILES[0], pm.BOOTSTRAP_PERCENTILES[1],
            result["spell_bootstrap_low"], result["spell_bootstrap_high"],
            result["n_gt_spells"], result["bootstrap_n"]))
    headline = result["spell_f1"][result["spell_iou_headline"]]
    print("      IoU {:.1f}   P {:.4f}  R {:.4f}  F1 {:.4f}   matched {}  missed {}  "
          "spurious {}".format(
              result["spell_iou_headline"], headline["precision"], headline["recall"],
              headline["f1"], headline["n_matched"], headline["n_missed"],
              headline["n_spurious"]))

    print("  alive/dead F1 {:.4f}   team acc  gt-alive {:.4f}  both-alive {:.4f}".format(
        result["alive_dead_f1"], result["team_acc_gt_alive"], result["team_acc_both_alive"]))
    print("  share        gt home {:.4f}  pred {}  err {}".format(
        result["gt_home_share"],
        "{:.4f}".format(result["pred_home_share"]) if result["share_defined"] else "--",
        "{:.4f}".format(result["share_err"]) if result["share_defined"]
        else "undefined (no alive frames predicted)"))
    print("  fragmentation  {} / {} = {:.2f}".format(
        result["n_pred_segments"], result["n_gt_segments"], result["fragmentation"]))
    print("  abstained    {} frames of X on graded ground truth ({:.2f}%)".format(
        result["n_abstain_on_graded"],
        100.0 * result["n_abstain_on_graded"] / max(1, result["n_graded"])))


def truncate(name):
    if len(name) <= NAME_WIDTH:
        return name
    return name[:NAME_WIDTH - 1] + "~"


def signed(value):
    """'+4.2' / '-3.0' / '--', so a lag reads as a lag at a glance."""
    if value is None:
        return "--"
    return "{:+.1f}".format(value)


def scoreboard_row(rank, entry, taus):
    """Flat dict for one line of scoreboard.csv, scored or skipped."""
    row = {"rank": rank, "submission": entry["name"],
           "status": "skipped" if entry["result"] is None else "scored",
           "reason": entry.get("reason") or ""}
    if entry["result"] is None:
        return row

    result = entry["result"]
    # Named for the board rather than for evaluate()'s internals, because these are the two a
    # reader arrives looking for after seeing them on screen.
    row["poss"] = result["n_pred_spells"]
    row["dur"] = result["mean_spell_duration_s"]
    for key in ("macro_f1", "spell_f1_mean", "spell_f1_at_headline", "spell_iou_headline",
                "spell_bootstrap_low", "spell_bootstrap_high",
                "n_gt_spells", "spell_count_error", "gt_mean_spell_duration_s",
                "bootstrap_low", "bootstrap_high", "kappa", "accuracy",
                "alive_dead_f1", "team_acc_gt_alive", "team_acc_both_alive",
                "gt_home_share", "pred_home_share", "share_err", "share_defined",
                "n_gt_segments", "n_pred_segments", "fragmentation",
                "n_graded", "n_abstain_on_graded"):
        row[key] = result[key]
    for name in pm.SCORED:
        row["f1_" + name] = result["per_class"][name]["f1"]
        row["precision_" + name] = result["per_class"][name]["precision"]
        row["recall_" + name] = result["per_class"][name]["recall"]

    loose = result["transitions"][taus[-1]]
    for key in ("mean_latency", "median_latency", "mean_abs_latency", "p90_abs_latency",
                "n_matched", "n_missed", "n_spurious", "n_ambiguous"):
        row[key] = loose[key]
    for tol in taus:
        transition = result["transitions"][tol]
        row["trans_f1_{}".format(tol)] = transition["f1"]
        row["trans_precision_{}".format(tol)] = transition["precision"]
        row["trans_recall_{}".format(tol)] = transition["recall"]
    return row


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return "{:.6f}".format(value)
    return str(value)


def write_csv(path, header, rows):
    lines = [",".join(header) + "\n"]
    for row in rows:
        lines.append(",".join(format_value(row.get(col)) for col in header) + "\n")
    with open(path, "w") as handle:
        handle.writelines(lines)


def write_transitions_csv(path, result, loose_tau):
    """Every transition, matched or not, with video frames in it.

    This is the artifact that makes an error watchable: pred_frame is a video frame, so a
    spurious transition can be taken straight to that frame of the video to see what fooled it.
    """
    header = ("status", "gt_frame", "pred_frame", "latency",
              "gt_from", "gt_to", "pred_from", "pred_to")
    transition = result["transitions"][loose_tau]
    rows = []
    for gt, pred in transition["pairs"]:
        rows.append({"status": "matched", "gt_frame": gt[0], "pred_frame": pred[0],
                     "latency": pred[0] - gt[0],
                     "gt_from": pm.LABELS[gt[1]], "gt_to": pm.LABELS[gt[2]],
                     "pred_from": pm.LABELS[pred[1]], "pred_to": pm.LABELS[pred[2]]})
    for gt in transition["missed"]:
        rows.append({"status": "missed", "gt_frame": gt[0],
                     "gt_from": pm.LABELS[gt[1]], "gt_to": pm.LABELS[gt[2]]})
    for pred in transition["spurious"]:
        rows.append({"status": "spurious", "pred_frame": pred[0],
                     "pred_from": pm.LABELS[pred[1]], "pred_to": pm.LABELS[pred[2]]})
    rows.sort(key=lambda r: (r.get("gt_frame") if r.get("gt_frame") is not None
                             else r.get("pred_frame"), r["status"]))
    write_csv(path, header, rows)


def json_safe(result):
    """Drop the transition tuples, which are for the per-submission transitions CSV instead."""
    out = {k: v for k, v in result.items() if k != "transitions"}
    out["transitions"] = {}
    for tol, transition in result["transitions"].items():
        out["transitions"][str(tol)] = {
            k: v for k, v in transition.items()
            if k not in ("pairs", "missed", "spurious")}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", default=DEFAULT_GT, metavar="PUBLIC|PRIVATE|PATH",
                        help="'public' for the 5-minute ground truth in ./gt, 'private' for the "
                             "full match, or a path to a CSV (default: {})".format(DEFAULT_GT))
    parser.add_argument("--results", default=RESULTS_DIR,
                        help="directory of submissions (default: ./results)")
    parser.add_argument("--out", default=OUT_DIR,
                        help="where to write scoreboard.csv and reports/ (default: here)")
    parser.add_argument("--tau", type=int, nargs="+", default=list(pm.TAUS),
                        help="transition tolerances in frames (default: {})".format(
                            " ".join(str(t) for t in pm.TAUS)))
    parser.add_argument("--bootstrap", type=int, default=pm.BOOTSTRAP_N,
                        help="segment-bootstrap draws for the macro-F1 interval, 0 = off")
    parser.add_argument("--no-detail", action="store_true",
                        help="print the ranked table only")
    args = parser.parse_args()

    taus = tuple(sorted(set(args.tau)))
    loose_tau = taus[-1]

    gt_path = find_gt(args.gt)
    meta = load_meta(gt_path)
    gt_codes = pm.label_codes(pm.read_label_csv(gt_path))
    expected = meta.get("n_video_frames")
    if expected is not None and len(gt_codes) != expected:
        # Fatal: a wrong ground truth invalidates every score, not just one submission.
        raise ValueError("ground truth has {} rows but its metadata says {}".format(
            len(gt_codes), expected))

    paths = sorted(glob.glob(os.path.join(args.results, "*.csv")))
    if not paths:
        raise IOError(
            "no *.csv in {}\nPut your submission there. To see the scorer work with nothing of "
            "your own yet:\n    python starter/make_dummy_submission.py".format(args.results))

    scored, skipped = [], []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        result, reason = score_submission(path, gt_codes, taus, args.bootstrap)
        entry = {"name": name, "path": path, "result": result, "reason": reason}
        (skipped if result is None else scored).append(entry)

    scored.sort(key=lambda e: rank_key(e, loose_tau))
    skipped.sort(key=lambda e: e["name"])

    print_scoreboard(scored, skipped, gt_path, meta, loose_tau)
    if not args.no_detail:
        for entry in scored:
            print_detail(entry, taus)

    reports_dir = os.path.join(args.out, REPORTS_DIRNAME)
    if not os.path.isdir(reports_dir):
        os.makedirs(reports_dir)

    header = scoreboard_columns(taus)
    rows = [scoreboard_row(rank, entry, taus) for rank, entry in enumerate(scored, start=1)]
    rows += [scoreboard_row(None, entry, taus) for entry in skipped]
    scoreboard_path = os.path.join(args.out, "scoreboard.csv")
    write_csv(scoreboard_path, header, rows)

    for entry in scored:
        report = json_safe(entry["result"])
        report["submission"] = entry["name"]
        report["ground_truth"] = os.path.basename(gt_path)
        report["taus"] = list(taus)
        with open(os.path.join(reports_dir, entry["name"] + ".json"), "w") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        write_transitions_csv(
            os.path.join(reports_dir, entry["name"] + "_transitions.csv"),
            entry["result"], loose_tau)

    print("\nWrote {}".format(scoreboard_path))
    print("Wrote {} per-submission reports to {}".format(len(scored), reports_dir))
    if skipped:
        print("\n{} submission(s) skipped; see the reasons above.".format(len(skipped)))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
