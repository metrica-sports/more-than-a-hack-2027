# The challenge

## The task

Given one full football match — two videos, automatic tracking with the ball, automatic tactical
tagging, homography — produce one label for every frame, saying which team is in possession or that
the ball is not in play.

199,319 frames. 25 frames per second. Roughly two hours of video, of which about 80 minutes is
graded.

That is the entire problem of Phase 1. The submission file is scored directly, 
but you will also provide the code and technical documentation explaining how you
implemented your proposal.

---

## The four labels

| label | meaning |
|---|---|
| `H` | **home** team in possession, ball alive |
| `A` | **away** team in possession, ball alive |
| `D` | **dead ball** — the ball is not in play |
| `X` | **no answer** — your system declines to guess, or the frame has no ground truth |

**`X` is a real option and it is not a trick.** Frames the ground truth does not cover — before
kickoff, halftime, after the final whistle — are never scored, so an `X` there costs nothing.

On graded frames the two metrics charge it differently, and the difference is worth knowing before
you design around it. `macroF1` counts an `X` as a miss, exactly like a wrong answer. `spellF1` —
the metric the board ranks on — never sees it: the scorer fills an abstention with the last label
you gave in that half before it matches possessions. So on the ranking metric an `X` costs whatever
repeating your previous answer costs — nothing in the middle of a possession, a great deal across a
hand-over, where your old label bleeds into the new state.
[EVALUATION.md](EVALUATION.md) has the measured cost of each case.

What it is genuinely for: a system that knows it has lost the ball can say so instead of inventing
a label, and the scorer treats a gap inside a real possession as a gap rather than as two
possessions with a transition in the middle.

**Dead-ball ownership is deliberately not asked for.** The tracking names a team even while the
ball is out of play — whoever will restart, or touched it last — and the ground truth throws that
away, because inferring who "owns" a dead ball is a different and much more ambiguous problem. When
the ball is dead the answer is `D`.

---

## Possession rules

> ### ⚠️ TBD — the operational definition
>
> This section will define exactly how the ground truth treats the ambiguous cases:
>
> - **passes** — when does possession leave the passer
> - **challenges and 50/50s** — who owns a contested ball, and for how long
> - **rebounds and deflections** — off a defender, off the keeper, off the post
> - **restarts** — the boundary between dead and alive at throw-ins, corners, goal kicks and free
>   kicks, and who is in possession at the moment of the restart
> - **temporarily loose balls** — a ball rolling free after a tackle, before anyone reaches it
>
> These rules decide a meaningful fraction of the frames near every transition, which is precisely
> where the metrics are sensitive. They will be published here before the competition opens.
>
> **Until then, treat the 5-minute public ground truth as the specification.** It is a worked
> example of every rule, frame by frame, for 17 possessions. If your reading of the data disagrees
> with it, the ground truth is right — and telling us about the disagreement is useful.

### What is already settled

- **Three states, not more.** Every graded frame is `H`, `A` or `D`. There is no "contested", no
  "transition" and no "unknown" in the ground truth.
- **A possession is a maximal run of one team with the ball alive.** `H H H A A` is two
  possessions. `H H D D H H` is two possessions with a stoppage between them, not one. This is the
  definition the ranking metric is built on, so it is worth internalising.
- **Dead ball is a stoppage, never a possession.**
- **The ground truth is derived from corrected tracking data** — not from a human watching video,
  and not from event data. The derivation is mechanical: a frame is `D` where the corrected
  tracking marks the ball dead, and otherwise takes the team the tracking names as owning it.
  Nothing is smoothed, interpolated or hand-adjusted afterwards.

  So the question that matters is not *how* the labels are computed, but what the corrected
  tracking means by "dead" and by "owning" — which is what the section above will define.

---

## Scope: what this challenge is not

It is **not** a multi-object tracking problem, and it is **not** game-state reconstruction.

The tracking, tagging and homography are given to you so that nobody has to rebuild detection,
re-identification and camera calibration before reaching the actual question: *what does the
geometry of twenty-two players and a ball tell you about who is in control?*

Everything handed to you is **automatic, not corrected**. The tracking has gaps, fragmented tracks
and identity switches, carries no player identifiers, and **the ball is missing on 41% of graded
frames**. The Smart Tagging is inferred too. Handling that noise is part of the problem, not an
obstacle to it — see [DATA.md](DATA.md).

Improving the provided data, or replacing it with your own detection and tracking, is entirely
legitimate. It is a starting point, not a constraint.

---

## What you may use

**Any method**, provided the organisers can access and inspect it and your technical document
explains it. Any language, framework, model or external data; LLMs commercial or open.

**What you may not do is annotate the data by hand** — the labels must come out of an algorithm.
Full rules in **[RULES.md](RULES.md)**.

---

## Deliverables and scoring

Three things through one form — the possession file, a public code repository, and a short
technical document. See **[SUBMISSION.md](SUBMISSION.md)**.

The score decides the ranking; the repository and document are what make it mean something. Ranked
on **`spellF1`**, with **`macroF1`** reported beside it as the first tie-break. Those two disagree
often and on purpose, and understanding why is most of the work —
**[EVALUATION.md](EVALUATION.md)** demonstrates it with a baseline you can run.
