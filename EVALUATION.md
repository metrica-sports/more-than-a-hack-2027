# How you are scored

The ranking metric is not accuracy, and it is not per-frame F1. 
A system tuned for either can achieve excellent per-frame scores while producing a completely unrealistic number of possessions — and still rank poorly.
Knowing that now is worth more than any amount of tuning later.

---

## Run it yourself

```bash
python evaluation/run_evaluation.py
```

Scores every `*.csv` in `evaluation/results/` against the **public** 5-minute ground truth and
prints a ranked board. On a fresh clone that is the four baselines, so it works before you have
written anything.

`--gt` takes a **name**, not a path: `public` (the default, ships here), `private` (the full match,
organisers only), or a path to any `frame,label` CSV.

A submission that fails validation is **skipped, not fatal** — it appears at the bottom with the
reason, so one broken file cannot hide everyone else's results.

**This is the file that produces the official result.** Organisers run this exact script against
the private full-match ground truth; the published board is its output. There is no stricter
version held back.

---

## The two headline numbers

### macroF1 — right label, right frame

The mean of the per-class F1 scores for `H`, `A` and `D`, over every graded frame. Frame-exact.

Averaging the three classes unweighted rather than counting frames matters, because dead ball is a
large share of all frames. Plain accuracy would give an "always dead ball" guess something that
reads like a real result; macroF1 gives it a score near the floor, which is the honest answer.

- **The floor is not 0.** A constant guess still scores something, so the usable range is roughly
  `[0.16, 1.0]` rather than `[0, 1]`.
- **Differences below ~0.01 are noise.** A match has a great many graded frames but far fewer
  segments, and the segment count is the effective sample size.

### spellF1 — right possessions

**This is what the board is ranked on.**

Each possession is treated as one object rather than a few hundred frames. A possession is a
maximal run of `H` or of `A`; dead-ball stretches are stoppages, not possessions.

Predicted possessions are matched one-to-one against true ones by **temporal IoU**, and the F1 of
that matching is averaged over thresholds 0.3, 0.4, 0.5, 0.6 and 0.7 — COCO's mAP trick applied to
time instead of space.

It exists because **no per-frame metric can represent possession**: the overwhelming majority of
graded frames sit in the middle of a possession where nothing is happening, dead ball takes a far
larger share of frames than of segments, and the longest possession of a match counts hundreds of
times the shortest even though a coach counts each one once.

Range is a true `[0, 1]` — a constant guess scores exactly 0.

| case | spellF1 |
|---|---|
| perfect | 1.0000 |
| perfect but 1 frame late | **1.0000** — a one-frame offset is free |
| perfect but 3 / 8 frames late | 0.9818 / 0.8933 |
| perfect but 1 s late | 0.6539 |
| random labels, or any constant guess | 0.0000 |

**Differences below ~0.04 are noise** on the full match — 3–4× macroF1's band, because the
effective sample is the match's possessions scored hit-or-miss rather than every frame. The gaps
between submissions widen in proportion, so it is not a worse ranking key.

---

## How an abstention is scored

`X` means two different things to the two metrics, and the difference decides where it is worth
writing one.

- **`macroF1`** charges every `X` on a graded frame as a miss, exactly like a wrong answer.
- **`spellF1`** never sees it. Before any temporal metric runs, the scorer forward-fills each `X`
  with the last non-`X` label in the same half, back-filling at the head of a half. By the time
  possessions are matched there are no abstentions left in the array.

So on the ranking metric `X` does not mean "no answer". It means **repeat my previous answer**, and
it costs whatever that costs.

What follows from that: going quiet through an occlusion **inside** a possession is free on the
ranking metric, and safer than a guess that might split the possession in two. At a transition it is
the opposite — guess, because the fill will otherwise carry your previous answer across the
hand-over.

---

## See the disagreement for yourself

The four baselines in `evaluation/results/` are **the same algorithm four times**. Nearest tracked
player to the ball, then a majority vote over a window. The only thing that changes is the window.

Scored on the full match:

| baseline | macroF1 | spellF1 |
|---|---|---|
| `baseline_raw` | 0.6532 | 0.0486 |
| `baseline_smooth25` | **0.6752** | 0.1996 |
| `baseline_smooth50` | 0.6724 | **0.2433** |
| `baseline_smooth125` | 0.6356 | 0.2242 |

`macroF1` moves by 0.04 across the entire sweep and picks `smooth25`. `spellF1` moves by a factor
of five and picks `smooth50`. Regenerate the files with `python starter/baseline.py`, score them
against the public window yourself, or walk through it in `starter/03_first_submission.ipynb`.

**`baseline_smooth50` is the published floor: spellF1 0.2433.** Beat it.

The possession counts and durations behind these scores are not published, because they would hand
you the ground truth's own counts to tune against. Your submission's counts are yours to inspect —
`validate_submission.py` prints them, and the board shows them next to the truth's whenever you
score against a ground truth you hold.

---

## The public window is not a leaderboard

The ground truth in this repository covers **5 minutes — 17 possessions**, and that is the whole
sample you can measure yourself against. The baselines show exactly what that costs:

| baseline | **public** spellF1 | **full match** spellF1 |
|---|---|---|
| `baseline_smooth50` | 0.3256 | **0.2433** ← actually better |
| `baseline_smooth125` | **0.4242** ← looks better | 0.2242 |

**Tuning the smoothing window against the public window picks the wrong one** — and that is a
single hyperparameter on a trivial baseline, with nothing adversarial about it. One missed
possession moves `spellF1` by roughly six points here.

Use the public window to check that your pipeline is *correct* — frames aligned, file well-formed,
labels meaning what you think — never to choose between two versions of it. When a gap looks real,
run `--bootstrap 1000` and check whether the intervals overlap. They usually will.

---

## The definitions are in the code

`evaluation/possession_metrics.py` is the specification. Every number on the board is defined
there, in about 720 lines of numpy with the reasoning in the docstrings — including why macro-F1
rather than accuracy, why transition matching is one-to-one rather than nearest-neighbour, and how
the noise floors were measured.

If this document and that file ever disagree, the file is right.
