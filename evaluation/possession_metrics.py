"""Metrics for per-frame possession inference, scored against a H/A/D/X ground truth.

numpy and pandas only, on purpose: no sklearn, nothing that pins a Python version, and
hand-rolling a 4x4 confusion matrix is fifteen lines. Keeping it dependency-free means the scorer
runs anywhere, and produces identical output under 3.8 and 3.11.

THE RANKING METRIC is spell_f1_mean -- possessions matched one-to-one by temporal IoU, defined
further down. macro-F1 over {H, A, D} is reported beside it as the per-frame view and is the first
tie-break. It is averaged unweighted over the frames where the ground truth is not X, rather than
counting frames, which is what stops a degenerate always-one-label submission from scoring
respectably, which is exactly what plain accuracy would do. It is however nearly blind to a uniform timing offset: lagging every
transition by ten frames barely moves it, because only a small fraction of frames sit near a
boundary. That is why the transition metrics below are computed as well, not as extras.

READ THIS BEFORE COMPARING TWO SCORES. A match has a great many frames but far fewer segments, so
consecutive frames are nowhere near independent and the effective sample size is the segment
count, not the frame count. One misjudged long dead-ball stretch moves macro-F1 by about a
percentage point on its own, so macro-F1 gaps below roughly 0.01 are segment-level noise rather
than signal. Set BOOTSTRAP_N to get a block-bootstrap interval over segments, which is the honest
way to answer "is this gap real"; do not compute per-frame intervals, they come out an order of
magnitude too narrow.

Usage:
    import possession_metrics as pm
    gt = pm.label_codes(pm.read_label_csv("gt/BAR-ATH_possession_gt.csv"))
    pred = pm.label_codes(pm.read_label_csv("results/mine.csv"))
    print(pm.evaluate(gt, pred)["macro_f1"])
"""

import numpy as np
import pandas as pd

# Params
# X is both "no ground truth here" (in the GT) and "I abstain" (in a submission). One symbol for
# both because the grading rule is the same either way: GT X is never scored, and a predicted X
# on a graded frame is a miss.
LABELS = ("H", "A", "D", "X")
SCORED = ("H", "A", "D")
H, A, D, X = 0, 1, 2, 3
N_LABELS = len(LABELS)
N_SCORED = len(SCORED)

# Tolerances for transition matching, in frames at 25 fps: ~0.5 s and 1.0 s. Reported as a pair
# because the gap between them is a lag detector -- a submission that scores badly at 12 and well
# at 25 has a systematic delay in that range, which no frame-level metric reveals.
TAUS = (12, 25)

# A possession is a team spell with the ball alive. D runs are stoppages, not possessions, so
# spells are maximal runs of H or of A. Dead-ball detection is covered separately by
# alive_dead_f1 and by the D row of the per-class table.
POSSESSION_LABELS = (H, A)
# IoU thresholds for possession-spell matching, averaged mAP-style. Averaging over IoU is the
# temporal action detection convention and is principled here, because IoU is a real quality
# axis for an interval. (Averaging over *frame tolerances* is not, and was rejected: the
# frame metric already costs only 0.0031 for a uniform 1-frame lag, below the noise floor.)
#
# Not called mAP anywhere on purpose: submissions are hard labels with no confidence scores, so
# there is no precision-recall curve to integrate. This is a mean F1 over IoU thresholds, and the
# mAP analogy is to the averaging axis alone.
SPELL_IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
SPELL_IOU_HEADLINE = 0.5
FPS = 25  # only used to report spell durations in seconds, which is how coaches read them

# Typical 5-95 percentile width of spell_f1_mean under a bootstrap over the ground-truth
# possessions of a full match, so a gap smaller than this is not a result. Measured across a set
# of reference submissions spanning realistic failure modes. It is 3-4x the ~0.01 floor on
# macro_f1, because the effective sample is the possessions scored hit-or-miss rather than every
# frame -- but the gaps between submissions widen in proportion, so exactly as many adjacent
# pairs stay separable under either metric.
SPELL_NOISE_BAND = 0.04

# Block bootstrap draws over GT segments. 0 = off (fast); 1000 for a final leaderboard.
BOOTSTRAP_N = 0
BOOTSTRAP_SEED = 0
BOOTSTRAP_PERCENTILES = (5, 95)

# Sentinel for possession share when a submission predicts no alive frames at all. Paired with
# share_defined in the output so it can never be mistaken for a measurement.
SHARE_UNDEFINED = 1.0


# -- reading and validation --------------------------------------------------------------------

def read_label_csv(path):
    """Read a frame,label CSV and return the label column as an object array.

    Tolerant about header case and surrounding whitespace, strict about everything else: the
    validation lives in validate_submission so a bad file can be reported rather than raised.
    """
    frame = pd.read_csv(path, dtype={"label": "object"})
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "label" not in frame.columns or "frame" not in frame.columns:
        raise ValueError("expected columns 'frame' and 'label', got {}".format(
            list(frame.columns)))
    return frame


def validate_submission(frame, n_expected):
    """Return None if the table is a usable submission, else a one-line reason why it is not."""
    if len(frame) != n_expected:
        return "row count {} != {}".format(len(frame), n_expected)

    frames = frame["frame"].values
    if not np.issubdtype(frames.dtype, np.integer):
        return "'frame' column is not integer (dtype {})".format(frames.dtype)
    if not np.array_equal(frames, np.arange(n_expected)):
        # One comparison covers 0-based, monotonic, contiguous and duplicate-free.
        return "'frame' column is not 0..{} in order".format(n_expected - 1)

    labels = frame["label"].values
    blank = pd.isna(labels)
    if blank.any():
        # Never coerce a blank cell to X: an empty field is a mistake, while X is a deliberate
        # abstention, and scoring one as the other would hide the mistake.
        return "empty label in {} rows; write X to abstain".format(int(blank.sum()))

    cleaned = np.array([str(v).strip().upper() for v in labels], dtype=object)
    bad = sorted(set(cleaned) - set(LABELS))
    if bad:
        return "labels outside {}: {}".format(list(LABELS), bad[:5])
    return None


def label_codes(frame_or_labels):
    """Map label strings to 0..3. Accepts the DataFrame from read_label_csv or a bare array."""
    values = frame_or_labels
    if isinstance(values, pd.DataFrame):
        values = values["label"].values
    values = np.array([str(v).strip().upper() for v in values], dtype=object)

    codes = np.full(len(values), -1, dtype=np.int8)
    for index, name in enumerate(LABELS):
        codes[values == name] = index
    if (codes < 0).any():
        raise ValueError("labels outside {}: {}".format(
            list(LABELS), sorted(set(values[codes < 0]))))
    return codes


# -- frame-level metrics -----------------------------------------------------------------------

def confusion(gt_codes, pred_codes):
    """4x4 counts, rows = ground truth, cols = prediction."""
    pair = gt_codes.astype(np.int64) * N_LABELS + pred_codes.astype(np.int64)
    return np.bincount(pair, minlength=N_LABELS * N_LABELS).reshape(N_LABELS, N_LABELS)


def scored_confusion(cm):
    """The 3x4 slice everything is scored on: GT rows H/A/D only, all four prediction columns.

    The whole X policy is this slice. False positives are summed over three GT rows, so a frame
    the ground truth does not cover can never manufacture one. False negatives are summed over
    four prediction columns, so a predicted X on a graded frame does count as a miss. No
    special-casing anywhere else.
    """
    return cm[:N_SCORED, :]


def per_class_prf(cm3, k):
    """(precision, recall, f1) for class k from the 3x4 scored confusion."""
    true_positive = float(cm3[k, k])
    false_positive = float(cm3[:, k].sum()) - true_positive
    false_negative = float(cm3[k, :].sum()) - true_positive

    precision = true_positive / (true_positive + false_positive) \
        if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) \
        if true_positive + false_negative else 0.0
    if precision + recall == 0.0:
        return 0.0, 0.0, 0.0  # absent class or nothing predicted -> 0, never nan
    return precision, recall, 2.0 * precision * recall / (precision + recall)


def macro_f1(cm3):
    """Unweighted mean F1 over H, A and D. The metric the scoreboard ranks on."""
    return float(np.mean([per_class_prf(cm3, k)[2] for k in range(N_SCORED)]))


def accuracy(cm3):
    """Fraction of graded frames labelled correctly. Predicted X counts against."""
    total = float(cm3.sum())
    if not total:
        return 0.0
    return float(sum(cm3[k, k] for k in range(N_SCORED))) / total


def cohens_kappa(cm3):
    """Chance-corrected agreement. Exactly 0.0 for any constant-guess submission.

    Reported as a secondary column only. Its chance level is computed from the submission's own
    marginals, so shifting your predicted class proportions changes your score without improving
    any single decision -- a bad property to rank a leaderboard on.
    """
    square = np.zeros((N_LABELS, N_LABELS), dtype=np.float64)
    square[:N_SCORED, :] = cm3  # the unused GT X row stays zero and drops out of both terms
    total = square.sum()
    if not total:
        return 0.0

    observed = np.trace(square) / total
    expected = sum(square[k, :].sum() * square[:, k].sum()
                   for k in range(N_LABELS)) / total ** 2
    if expected >= 1.0:
        return 0.0
    return float((observed - expected) / (1.0 - expected))


def alive_dead_f1(cm3):
    """F1 of the alive/dead call alone, with ALIVE as the positive class.

    ALIVE and not DEAD because dead ball is a large share of all frames: with DEAD positive, an
    always-dead submission scores well on this and reads like competence. With ALIVE positive it
    scores 0, which is the honest answer.
    """
    true_positive = float(cm3[:D, :D].sum())              # GT alive, predicted alive
    false_positive = float(cm3[D, :D].sum())              # GT dead, predicted alive
    false_negative = float(cm3[:D, D:].sum())             # GT alive, predicted dead or X

    precision = true_positive / (true_positive + false_positive) \
        if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) \
        if true_positive + false_negative else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def team_accuracy(cm3):
    """(strict, both_alive) H-vs-A accuracy.

    strict is over every GT-alive frame, so answering D or X there counts as wrong. both_alive is
    over the frames where GT and prediction agree the ball is alive, i.e. the team call in
    isolation, with the alive/dead call already charged by alive_dead_f1.
    """
    correct = float(cm3[H, H] + cm3[A, A])
    gt_alive = float(cm3[:D, :].sum())
    both_alive = float(cm3[:D, :D].sum())
    return (correct / gt_alive if gt_alive else 0.0,
            correct / both_alive if both_alive else 0.0)


def possession_share(cm3):
    """(gt_share, pred_share, abs_err, defined) home share of alive time.

    Each side is measured over its own alive frames, because this is the number a submission
    would actually report as "the home side had 55% of the ball" -- not a per-frame agreement.
    """
    gt_alive = float(cm3[:D, :].sum())
    pred_alive = float(cm3[:, :D].sum())
    gt_share = float(cm3[H, :].sum()) / gt_alive if gt_alive else 0.0
    if not pred_alive:
        return gt_share, 0.0, SHARE_UNDEFINED, False
    pred_share = float(cm3[:, H].sum()) / pred_alive
    return gt_share, pred_share, abs(pred_share - gt_share), True


# -- temporal structure ------------------------------------------------------------------------

def valid_runs(gt_codes):
    """Half-open [start, stop) spans where the ground truth is not X.

    Derived from the data, never hard-coded: for BAR-ATH this returns the two halves,
    (11185, 90652) and (110808, 190198). Everything temporal is computed inside a run and never
    across one, so the 20156-frame halftime hole cannot produce a phantom transition. That
    matters even when the labels either side happen to match -- here they are both D, so a
    whole-array run-length encoding silently merges them and undercounts the segments by one.
    """
    graded = (gt_codes != X).astype(np.int8)
    edges = np.diff(np.concatenate(([0], graded, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def fill_abstentions(pred_codes, runs):
    """Copy of pred_codes with X forward-filled (then back-filled at the run head) inside runs.

    Frame-level metrics see the raw X and charge it. The temporal metrics see this filled version
    instead, because 'H X X H' is one uninterrupted possession that the model went quiet during,
    not two possessions with a pair of transitions in between. Without this, an abstention is
    charged twice: once as a frame-level miss and again as two spurious transitions.
    """
    filled = pred_codes.copy()
    for start, stop in runs:
        window = filled[start:stop]
        known = np.flatnonzero(window != X)
        if not len(known):
            continue  # a run the submission abstained on entirely; nothing to carry forward
        # For each position, the index of the most recent non-X at or before it, then clamp the
        # leading positions to the first known value (the back-fill).
        carry = np.maximum.accumulate(np.where(window != X, np.arange(len(window)), -1))
        carry[carry < 0] = known[0]
        filled[start:stop] = window[carry]
    return filled


def segments(codes, runs):
    """[(start, stop, code)] constant-label runs, encoded inside each valid run separately."""
    out = []
    for start, stop in runs:
        window = codes[start:stop]
        if not len(window):
            continue
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for i in range(len(bounds) - 1):
            out.append((start + int(bounds[i]), start + int(bounds[i + 1]),
                        int(window[bounds[i]])))
    return out


def transitions(codes, runs):
    """[(frame, from_code, to_code)] label changes, never spanning a run boundary.

    frame is the first frame of the new label, so it is directly usable as a seek target in a
    video player.
    """
    out = []
    for start, stop in runs:
        window = codes[start:stop]
        for index in (np.flatnonzero(window[1:] != window[:-1]) + 1).tolist():
            out.append((start + int(index), int(window[index - 1]), int(window[index])))
    return out


def match_transitions(gt_trans, pred_trans, tol, typed="to"):
    """One-to-one match of predicted transitions to ground-truth ones within +/- tol frames.

    Returns (pairs, missed, spurious), pairs being [(gt_transition, pred_transition)].

    Matching is on the destination label by default. 'D -> H' and 'D -> A' at the same frame are
    materially different events -- you spotted the restart but gave it to the wrong team -- so the
    destination has to agree. The origin does not: a predicted 'A -> H' against a true 'D -> H'
    got the event right and only the preceding state wrong, and that preceding state is already
    fully charged at frame level. typed="pair" demands both, typed="none" neither.

    Both lists are frame-sorted and feasibility is an interval, so matching earliest-to-earliest
    within each key is maximum-cardinality (the standard interval exchange argument) as well as
    deterministic. Cross-run pairing needs no explicit guard: runs are 20156 frames apart and
    tol is at most a few dozen, so the interval test rejects it.

    Caveat worth knowing: maximum cardinality does not mean minimum total latency, and when
    several maximum matchings exist the reported mean latency can shift by a few frames. A
    handful of ground-truth segments are shorter than the loose tolerance, so at tol=25 the
    windows really can overlap. That is what n_ambiguous in transition_scores is for.
    """
    def key(transition):
        if typed == "pair":
            return (transition[1], transition[2])
        if typed == "none":
            return 0
        return transition[2]

    pairs, missed, spurious = [], [], []
    gt_by_key, pred_by_key = {}, {}
    for transition in gt_trans:
        gt_by_key.setdefault(key(transition), []).append(transition)
    for transition in pred_trans:
        pred_by_key.setdefault(key(transition), []).append(transition)

    for group in sorted(set(gt_by_key) | set(pred_by_key)):
        gts = gt_by_key.get(group, [])
        preds = pred_by_key.get(group, [])
        i = j = 0
        while i < len(gts) and j < len(preds):
            delta = preds[j][0] - gts[i][0]
            if delta < -tol:
                spurious.append(preds[j])  # too early for this or any later gt
                j += 1
            elif delta > tol:
                missed.append(gts[i])      # no remaining pred can reach back to it
                i += 1
            else:
                pairs.append((gts[i], preds[j]))
                i += 1
                j += 1
        missed.extend(gts[i:])
        spurious.extend(preds[j:])
    return pairs, missed, spurious


def transition_scores(gt_trans, pred_trans, tol, typed="to"):
    """Precision, recall, F1 and latency for transition detection at one tolerance.

    Precision and recall are reported separately and always should be: jitter shows up as recall
    near 1.0 with precision near 0, which a single F1 hides.
    """
    pairs, missed, spurious = match_transitions(gt_trans, pred_trans, tol, typed)
    true_positive, false_positive, false_negative = len(pairs), len(spurious), len(missed)

    precision = true_positive / float(true_positive + false_positive) \
        if true_positive + false_positive else 0.0
    recall = true_positive / float(true_positive + false_negative) \
        if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    latencies = np.array([pred[0] - gt[0] for gt, pred in pairs], dtype=np.float64)
    ambiguous = _count_ambiguous(gt_trans, pred_trans, tol, typed)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_gt": len(gt_trans),
        "n_pred": len(pred_trans),
        "n_matched": true_positive,
        "n_missed": false_negative,
        "n_spurious": false_positive,
        "n_ambiguous": ambiguous,
        "mean_latency": float(latencies.mean()) if len(latencies) else None,
        "median_latency": float(np.median(latencies)) if len(latencies) else None,
        "mean_abs_latency": float(np.abs(latencies).mean()) if len(latencies) else None,
        "p90_abs_latency": float(np.percentile(np.abs(latencies), 90)) if len(latencies) else None,
        "pairs": pairs,
        "missed": missed,
        "spurious": spurious,
    }


def _count_ambiguous(gt_trans, pred_trans, tol, typed):
    """GT transitions with more than one feasible prediction, i.e. where the match was a choice.

    When this is 0 the latency figures are exact; when it is large they are indicative only.
    """
    def key(transition):
        if typed == "pair":
            return (transition[1], transition[2])
        if typed == "none":
            return 0
        return transition[2]

    by_key = {}
    for transition in pred_trans:
        by_key.setdefault(key(transition), []).append(transition[0])
    for group in by_key:
        by_key[group] = np.array(sorted(by_key[group]))

    count = 0
    for transition in gt_trans:
        frames = by_key.get(key(transition))
        if frames is None:
            continue
        if int(np.sum(np.abs(frames - transition[0]) <= tol)) > 1:
            count += 1
    return count


def fragmentation(gt_codes, pred_codes, runs):
    """(n_gt_segments, n_pred_segments, ratio). A jitter detector: a flickering submission can
    hold a respectable macro-F1 while producing thirty times too many possessions."""
    n_gt = len(segments(gt_codes, runs))
    n_pred = len(segments(pred_codes, runs))
    return n_gt, n_pred, (n_pred / float(n_gt) if n_gt else 0.0)


# -- possession-spell level --------------------------------------------------------------------
#
# Why this level exists at all. Frame macro-F1 is the right per-frame metric, but no per-frame
# metric can represent possession, because the overwhelming majority of graded frames sit in the
# interior of a segment where nothing is happening. A jittery submission can hold a per-frame
# score near 0.97 while reporting well over half as many possessions again as the truth, each far
# too short. Per-frame scoring also over-weights dead ball, which is a much larger share of frames
# than of segments, and lets the longest possession of a match count hundreds of times the
# shortest. Treating each possession as one object fixes all three.

def possession_spells(codes, runs):
    """[(start, stop, code)] maximal H or A runs -- the possessions, stoppages excluded.

    Built on the gap-aware segments(), so no spell can ever straddle the halftime hole.
    """
    return [(a, b, k) for a, b, k in segments(codes, runs) if k in POSSESSION_LABELS]


def spell_overlaps(gt_spells, pred_spells):
    """[(iou, gt_index, pred_index)] for every same-team overlapping pair, sorted for matching.

    Computed once and filtered per threshold by match_spells: the IoU of a pair does not depend
    on the threshold, so recomputing it for each of SPELL_IOUS would be five times the work for
    identical results. Sorted by (-iou, gt_index, pred_index) so ties break the same way on
    every run and the matching is reproducible.
    """
    out = []
    for i, (gt_start, gt_stop, gt_code) in enumerate(gt_spells):
        for j, (pred_start, pred_stop, pred_code) in enumerate(pred_spells):
            if pred_code != gt_code or pred_stop <= gt_start or gt_stop <= pred_start:
                continue
            intersection = min(gt_stop, pred_stop) - max(gt_start, pred_start)
            union = max(gt_stop, pred_stop) - min(gt_start, pred_start)
            out.append((intersection / float(union), i, j))
    out.sort(key=lambda item: (-item[0], item[1], item[2]))
    return out


def matched_gt_flags(overlaps, n_gt, threshold):
    """Boolean array over ground-truth spells, True where one was matched at this IoU.

    Greedy on the IoU-descending list, marking both sides consumed, so every ground-truth spell
    is credited to at most one prediction and vice versa. Returned per spell rather than as a
    count because the bootstrap needs to resample individual possessions.
    """
    flags = np.zeros(n_gt, dtype=bool)
    matched_pred = set()
    for iou, gt_index, pred_index in overlaps:
        if iou < threshold:
            break  # the list is IoU-descending, so nothing later can qualify either
        if flags[gt_index] or pred_index in matched_pred:
            continue
        flags[gt_index] = True
        matched_pred.add(pred_index)
    return flags


def match_spells(overlaps, n_gt, n_pred, threshold):
    """(tp, fp, fn) from a one-to-one greedy match of spells at one IoU threshold."""
    true_positive = int(matched_gt_flags(overlaps, n_gt, threshold).sum())
    return true_positive, n_pred - true_positive, n_gt - true_positive


def spell_scores(gt_codes, pred_codes, runs, ious=SPELL_IOUS):
    """Possession-spell detection precision/recall/F1 at each IoU, plus the mean F1 over them."""
    gt_spells = possession_spells(gt_codes, runs)
    pred_spells = possession_spells(pred_codes, runs)
    overlaps = spell_overlaps(gt_spells, pred_spells)

    per_iou, f1s = {}, []
    for threshold in ious:
        true_positive, false_positive, false_negative = match_spells(
            overlaps, len(gt_spells), len(pred_spells), threshold)
        precision = true_positive / float(true_positive + false_positive) \
            if true_positive + false_positive else 0.0
        recall = true_positive / float(true_positive + false_negative) \
            if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_iou[threshold] = {"precision": precision, "recall": recall, "f1": f1,
                              "n_matched": true_positive, "n_spurious": false_positive,
                              "n_missed": false_negative}
        f1s.append(f1)

    return {
        "per_iou": per_iou,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "n_gt": len(gt_spells),
        "n_pred": len(pred_spells),
    }


def possession_stats(codes, runs):
    """What a coach reads off a possession signal: how many, how long, and the home share.

    Reported alongside the F1s because they need no metric literacy -- a predicted possession
    count next to the truth's states the failure more plainly than any score. Note the home share
    is a poor discriminator on its own: a jittery submission can get it almost exactly right while
    inventing hundreds of possessions, so it is reported and never ranked on.
    """
    spells = possession_spells(codes, runs)
    lengths = [b - a for a, b, _ in spells]
    alive = int(np.sum((codes == H) | (codes == A)))
    return {
        "n_spells": len(spells),
        "mean_duration_s": float(np.mean(lengths)) / FPS if lengths else 0.0,
        "home_share": float(np.sum(codes == H)) / alive if alive else 0.0,
    }


# -- bootstrap ---------------------------------------------------------------------------------

def bootstrap_macro_f1(gt_codes, pred_codes, runs, n_draws,
                       seed=BOOTSTRAP_SEED, percentiles=BOOTSTRAP_PERCENTILES):
    """Block-bootstrap interval for macro-F1, resampling whole GT segments.

    Segments, not frames, because frames inside a possession are not independent observations.
    Returns (low, high) or (None, None) when disabled.
    """
    if not n_draws:
        return None, None

    gt_segments = segments(gt_codes, runs)
    if not gt_segments:
        return None, None

    # Per-segment prediction histogram, so a draw is a matrix add rather than a rescan of 199k
    # frames: contributions[i, p] = frames of segment i the submission called p.
    gt_of_segment = np.array([code for _, _, code in gt_segments], dtype=np.int64)
    contributions = np.zeros((len(gt_segments), N_LABELS), dtype=np.int64)
    for index, (start, stop, _) in enumerate(gt_segments):
        contributions[index] = np.bincount(pred_codes[start:stop].astype(np.int64),
                                           minlength=N_LABELS)

    rng = np.random.RandomState(seed)
    scores = np.empty(n_draws, dtype=np.float64)
    for draw in range(n_draws):
        picks = rng.randint(0, len(gt_segments), len(gt_segments))
        cm3 = np.zeros((N_SCORED, N_LABELS), dtype=np.int64)
        np.add.at(cm3, gt_of_segment[picks], contributions[picks])
        scores[draw] = macro_f1(cm3)
    low, high = np.percentile(scores, percentiles)
    return float(low), float(high)


def bootstrap_spell_f1(gt_codes, pred_codes, runs, n_draws,
                       seed=BOOTSTRAP_SEED, percentiles=BOOTSTRAP_PERCENTILES):
    """Bootstrap interval for spell_f1_mean, resampling whole possessions.

    Possessions, not segments and not frames, because spell_f1_mean scores each possession as one
    hit-or-miss observation, and that count is the sample size governing its variance. This is the
    interval that matters once the board ranks on spellF1; bootstrap_macro_f1 above answers the
    same question for the frame-level metric.

    Returns (low, high) or (None, None) when disabled.
    """
    if not n_draws:
        return None, None

    gt_spells = possession_spells(gt_codes, runs)
    if not gt_spells:
        return None, None
    pred_spells = possession_spells(pred_codes, runs)
    overlaps = spell_overlaps(gt_spells, pred_spells)
    n_gt, n_pred = len(gt_spells), len(pred_spells)

    # Matched-or-not per possession per threshold, computed once. A draw is then a resample of
    # rows rather than a re-run of the matching.
    flags = np.vstack([matched_gt_flags(overlaps, n_gt, threshold)
                       for threshold in SPELL_IOUS])

    rng = np.random.RandomState(seed)
    scores = np.empty(n_draws, dtype=np.float64)
    for draw in range(n_draws):
        picks = rng.randint(0, n_gt, n_gt)
        f1s = []
        for row in flags:
            true_positive = int(row[picks].sum())
            false_negative = n_gt - true_positive
            # n_pred is fixed while the resampled tp moves, so for a submission predicting fewer
            # spells than the truth a draw can imply tp > n_pred. Clamp rather than go negative.
            false_positive = max(0, n_pred - true_positive)
            f1s.append(2.0 * true_positive / (2 * true_positive + false_positive + false_negative)
                       if true_positive else 0.0)
        scores[draw] = np.mean(f1s)
    low, high = np.percentile(scores, percentiles)
    return float(low), float(high)


# -- entry point -------------------------------------------------------------------------------

def evaluate(gt_codes, pred_codes, taus=TAUS, bootstrap_n=BOOTSTRAP_N, typed="to"):
    """Every metric for one submission. The only function run_evaluation.py needs.

    Touches the full-length arrays exactly twice: once for the confusion matrix, once for the
    run-length encoding behind the temporal metrics.
    """
    if len(gt_codes) != len(pred_codes):
        raise ValueError("length mismatch: gt {} vs pred {}".format(
            len(gt_codes), len(pred_codes)))

    cm = confusion(gt_codes, pred_codes)
    cm3 = scored_confusion(cm)

    runs = valid_runs(gt_codes)
    filled = fill_abstentions(pred_codes, runs)
    gt_trans = transitions(gt_codes, runs)
    pred_trans = transitions(filled, runs)

    n_gt_seg, n_pred_seg, frag = fragmentation(gt_codes, filled, runs)
    gt_share, pred_share, share_err, share_defined = possession_share(cm3)
    strict_team, both_alive_team = team_accuracy(cm3)
    boot_low, boot_high = bootstrap_macro_f1(gt_codes, pred_codes, runs, bootstrap_n)
    spell_boot_low, spell_boot_high = bootstrap_spell_f1(gt_codes, filled, runs, bootstrap_n)

    # Possession-spell level, on the abstention-filled prediction for the same reason the
    # transition metrics use it: 'H X X H' is one possession the model went quiet during, not
    # two possessions with a gap.
    spells = spell_scores(gt_codes, filled, runs)
    gt_stats = possession_stats(gt_codes, runs)
    pred_stats = possession_stats(filled, runs)

    result = {
        "macro_f1": macro_f1(cm3),
        # Co-headline, deliberately not folded into macro_f1: they answer different questions
        # ("right label" vs "right possessions") and their disagreement is the useful signal.
        "spell_f1_mean": spells["mean_f1"],
        "spell_f1_at_headline": spells["per_iou"][SPELL_IOU_HEADLINE]["f1"],
        "spell_iou_headline": SPELL_IOU_HEADLINE,
        "spell_bootstrap_low": spell_boot_low,
        "spell_bootstrap_high": spell_boot_high,
        "n_gt_spells": spells["n_gt"],
        "n_pred_spells": spells["n_pred"],
        "spell_count_error": spells["n_pred"] - spells["n_gt"],
        "mean_spell_duration_s": pred_stats["mean_duration_s"],
        "gt_mean_spell_duration_s": gt_stats["mean_duration_s"],
        "accuracy": accuracy(cm3),
        "kappa": cohens_kappa(cm3),
        "alive_dead_f1": alive_dead_f1(cm3),
        "team_acc_gt_alive": strict_team,
        "team_acc_both_alive": both_alive_team,
        "gt_home_share": gt_share,
        "pred_home_share": pred_share,
        "share_err": share_err,
        "share_defined": share_defined,
        "n_gt_segments": n_gt_seg,
        "n_pred_segments": n_pred_seg,
        "fragmentation": frag,
        "n_frames": int(len(gt_codes)),
        "n_graded": int(cm3.sum()),
        "n_abstain_on_graded": int(cm3[:, X].sum()),
        "bootstrap_low": boot_low,
        "bootstrap_high": boot_high,
        "bootstrap_n": bootstrap_n,
        "confusion": cm.tolist(),
        "per_class": {},
        "transitions": {},
        # The per-IoU breakdown is for the detail block and the JSON report, not the CSV: five
        # thresholds x three figures would add fifteen columns for what the mean already ranks.
        "spell_f1": spells["per_iou"],
        "gt_home_share_spells": gt_stats["home_share"],
    }
    for k, name in enumerate(SCORED):
        precision, recall, f1 = per_class_prf(cm3, k)
        result["per_class"][name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": int(cm3[k, :].sum()),
        }
    for tol in taus:
        result["transitions"][tol] = transition_scores(gt_trans, pred_trans, tol, typed)
    return result
