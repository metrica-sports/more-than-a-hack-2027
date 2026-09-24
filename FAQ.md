# FAQ

### Are the 5 minutes of ground truth the test set?

No. They are a worked example, so you can check your pipeline is correct before spending a week on
the wrong thing. Your score comes from the full-match ground truth, which stays private.

The public window cannot rank anybody — the baselines here prove it: the smoothing window that wins
on the public 5 minutes loses on the full match. See [EVALUATION.md](EVALUATION.md).

### Can I submit only the 5 minutes?

No. Exactly 199,319 rows, every time. Write `X` where you have no answer. Submitting a short file
is the most common way to score zero, which is why `validate_submission.py` exists.

### What if my system genuinely does not know?

Write `X`. Outside the graded window it is ignored. Inside it, `macroF1` charges it as a miss —
but `spellF1`, which decides the ranking, never sees it: the scorer fills an abstention with your
last label in that half before matching possessions.

So an `X` costs whatever repeating your previous answer costs. Going quiet through an occlusion
inside a possession is free on the ranking metric, and safer than a guess that might split the
possession in two. Going quiet across a hand-over is expensive, because your old label is what gets
carried forward. If your 60/40 guess is at a transition, guess. [EVALUATION.md](EVALUATION.md) has
the measured cost of each case.

### Which frames are actually graded?

The two halves: video frames **11,185–90,651** and **110,808–190,197**. Everything else — before
the match, halftime, after the final whistle — is ungraded, so `X` there costs nothing.

One trap worth knowing: the second half is graded from frame 110,808 but does not kick off until
**113,742**. Those 2,934 frames are graded and all of them are `D`. The first half, by contrast, is
graded from the kickoff itself. Full table in [DATA.md](DATA.md), or
`games.half_video_blocks("BAR-ATH")`.

### Why is the ball missing so often?

Because the tracking is automatic and the ball is small, fast and frequently occluded. It is absent
on **41% of graded frames**, and disproportionately on the interesting ones — crowded boxes, aerial
duels, tackles.

That is the central difficulty of the challenge rather than a defect in the data. See
[DATA.md](DATA.md).

### Why is my submission full of tiny possessions?

Because per-frame decisions flicker. The raw baseline reports several times more possessions than
the match actually had, while still looking respectable on `macroF1`.

A possession is an interval, not a sequence of independent frames. Temporal smoothing, segmentation
or a state model is the first real thing to add — `starter/03_first_submission.ipynb` walks through
the simplest version.

### Why is the ranking metric not accuracy, or macro F1?

Because the overwhelming majority of graded frames sit in the middle of a possession where nothing
is happening, and the longest possession of a match counts hundreds of times the shortest — though
a coach counts each one once. A system can score near 0.97 per frame while reporting far more
possessions than really happened.
[EVALUATION.md](EVALUATION.md) has the full argument.

### macroF1 is reported too. Which should I tune?

`spellF1` decides the ranking; `macroF1` breaks ties. They sit side by side because they disagree
informatively — a submission strong on one and weak on the other is telling you something specific
about its failure mode.

### Can I use the broadcast video and its tracking?

Yes, and it comes with its own ATD and homography. But it is on **its own timeline** — cuts,
replays, a different frame count — and no mapping to the tactical video is provided. Your
submission is indexed to the tactical video, so mapping back is your problem.

If your file ends up with ~164k rows, you indexed the broadcast by mistake.

### Can I use my own tracking instead of the provided ATD?

Yes, and it is a legitimate line of attack. The provided data exists so nobody *has* to rebuild
detection and calibration first — a starting point, not a constraint. If better tracking gets you a
better answer, that is a better answer.

### Can I use the Smart Tagging? Isn't that cheating?

Use it — that is why it is there. It is not ground truth and not derived from it; Metrica generates
it automatically from the same video and tracking you have, without manual review, so it contains
false positives and false negatives like any automatic output.

Useful, because every phase carries a team tag and every set piece is a restart. Just remember the
intervals are *phases*, not possessions.

### Can I use LLMs, pre-trained models, external data?

Yes, provided the organisers can access what you used and your technical document explains it. See
[RULES.md](RULES.md).

### Can I hand-label some frames?

To **train or validate** a model, yes. To **produce the submission**, no — the labels you submit
must come out of an algorithm. See [RULES.md](RULES.md).

### What happens if my file is malformed?

It is skipped with a one-line reason and scores nothing. It does not affect anyone else. This is
why you run `validate_submission.py` on the exact file you are about to upload.

### The frame count of the video I decoded is not 199,319.

Then your decode is not the reference decode, and your file will be rejected on row count — which
is what that check is for. Do not pad or trim to fit; work out why the counts differ.

### Why is the frame column there if it is just the row number?

Because it is redundant on purpose. It turns an off-by-one into a hard error at validation time
instead of a score silently wrong by one frame everywhere, which costs almost nothing on `macroF1`
and is very hard to spot by eye.

### My scores changed between two runs with the same file.

They should not — the scorer is deterministic and identical under Python 3.8 and 3.11. Please
report it. (`--bootstrap` is seeded and reproducible, but changing the number of draws changes the
intervals.)

### Where do I put my submissions?

`evaluation/results/`. Anything ending in `.csv` there is scored. The folder is gitignored apart
from the four committed baselines, so your own files stay out of commits.

### Something here is wrong or unclear.

Open an issue. Corrections to the documentation, the alignment or the metrics are genuinely useful.
If a rule is unclear or your approach sits near a line, ask before you build on it.
