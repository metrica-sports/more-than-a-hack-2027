# Rules

## What you may use

**Any method at all**, subject to two conditions:

1. **The organisers can access and inspect it.** Anything in your pipeline has to be something we
   can look at and run — your repository, a public model, a commercial API, an open dataset.
2. **Your technical document explains it.** Not a full paper; enough that a reader understands what
   you built and why.

Within that, nothing is off the table. Any language, any framework. Pre-trained models, commercial
or open. LLMs, hosted or local. External data. Your own detection, tracking or event-recognition
models. Classical methods with no learning at all.

Both conditions exist for the same reason: the score decides the ranking, and the repository and
document are what make the score mean something. A method nobody can access and nobody can
understand cannot be verified, and an unverifiable result is not a result.

**A private in-house model or an unavailable internal dataset is the one thing this rules out.** If
your approach depends on something you cannot share, ask before you build on it.

### The provided data is a starting point, not a constraint

You are given automatic tracking, tactical tagging and homography so that no team has to rebuild
detection, re-identification and camera calibration before starting on the actual question.

**Improving that data, or replacing it with your own, is entirely legitimate** and needs no
justification. Run your own detector, fix the tracking, build your own event recognition — if it
produces a better answer, it is a better answer. We do not care where the intermediate
representations come from.

---

## What you may not do

### 1. Annotate the data by hand

**The labels must be produced by an algorithm.** A human watching the video and writing down who
has the ball is not a submission, no matter how accurate it is.

This applies whether or not any answer key was involved. Hand-labelling by eye, hand-correcting an
algorithm's output frame by frame, and hand-labelling against a leaked ground truth are all the
same thing as far as this rule is concerned: the file did not come from a system, so there is no
system to evaluate and nothing that would work on the next match.

Small amounts of hand-labelled data used to *train or validate* a model are fine and normal. The
distinction is whether the submitted labels come out of a model or out of a person.

### 2. Use the private ground truth

You may not use the full-match ground truth, or anything derived from it, beyond the 5 minutes
published in this repository. That includes any other source of corrected possession data for this
fixture.

### 3. Submit a file your repository cannot regenerate

Running your repository against the provided data must reproduce your submission. Not
approximately — the same file.

---

## Eligibility and teams

**<TBD: team size.>**

**<TBD: eligibility — who may enter.>**

**<TBD: code of conduct.>**

---

## The data

The match assets are licensed for this competition only, and the terms you accept at enrolment are
the authority on what that permits. They are not restated here.

Two things follow from them that are easy to get wrong:

- **Do not redistribute the data.** That includes committing it to a public repository. Your
  submission repository has to be public, so make sure the data is not in it — this repository's
  `.gitignore` already excludes `data/`, and yours should too.
- **Do not publish derived data that would reconstruct it.** Frame-level tracking exports, clipped
  video, and similar.

The code and documentation in this repository are separately licensed — see [LICENSE](LICENSE).
That licence covers this repository only and says nothing about the match data.

---

## Questions

If a rule is unclear, or your approach sits near a line, **ask before you build on it**. Open an
issue. An answer before the deadline is worth more to everyone than a disqualification after it,
and a question that needed asking usually means the rule needed clarifying.
