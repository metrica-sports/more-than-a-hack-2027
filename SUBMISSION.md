# Submitting

## The possession file

A CSV with a header line and one row per video frame.

```
frame,label
0,X
1,X
...
11185,A
11186,A
...
199318,X
```

| requirement | why it is strict |
|---|---|
| header is exactly `frame,label` | case and surrounding whitespace are tolerated; anything else is not |
| exactly **199,319** data rows | one per frame of the tactical video |
| `frame` is `0 … 199318`, in order | redundant with row order **on purpose** — it turns an off-by-one into a hard error instead of a score that is quietly wrong by one frame everywhere |
| `label` is one of `H` `A` `D` `X` | stripped and upper-cased before checking, so `h` and ` H ` are fine |
| no blank cells | an empty field is a mistake and is rejected as one. **Write `X` to abstain** |

UTF-8, `\n` or `\r\n`, no index column. Anything `pandas.read_csv` will parse is fine.

Frames where the **ground truth** is `X` are never scored — halftime, before kickoff, after the
final whistle. On a graded frame an `X` counts as a miss under `macroF1`; under `spellF1`, which
decides the ranking, the scorer fills it with your last label in that half before matching
possessions, so it costs whatever repeating that label costs. See [EVALUATION.md](EVALUATION.md).

### Check it before you send it

```bash
python evaluation/validate_submission.py my_submission.csv
```

This runs **exactly** the checks the official scorer runs. A file that fails them is skipped at
scoring time with a one-line reason and scores nothing.


---

## The three deliverables

### 1. The possession file

As above. This is what is scored and what decides the ranking.

### 2. A public code repository

Everything needed to go from the provided data to your submission file: source, pinned
dependencies, installation and execution instructions, and the command that regenerates the file.

The bar is that someone with the data bundle and your repository can reproduce your file. Not
approximately — the same file.

Make sure the match data is not in it. Your repository is public and the data is not
redistributable.

### 3. A technical document

Short — a few pages is plenty. PDF or Markdown. It should cover:

- **the approach** — what you built, and why that rather than something else
- **which assets you used and how** — saying which ones you tried and *dropped* is just as useful
- **the possession rules you implemented** — where you drew the line on loose balls, challenges,
  restarts
- **limitations** — where it fails, and what you would fix with more time
- **reproduction** — how to run it

---

## How to submit

One Google Form, three fields plus your team name:

**<TBD: submission form URL>**

| field | what |
|---|---|
| team name | becomes the name on the published board |
| possession file | upload your `.csv` (~1.9 MB) |
| code repository | a URL. Public, or access granted to the organisers |
| technical document | upload a `.pdf` or `.md` |

**Deadline: <TBD>**

Four practical things, because a form is less forgiving than a folder:

- **Name the CSV after your team** — `<team_name>.csv`. The scorer names each row of the board
  after the file, so this is what puts your name on it.
- **You need a Google account** to upload. Worth discovering now rather than at the deadline.
- **One response per team.** You may edit it; the last response before the deadline is the one
  scored.
- **There is no live leaderboard and no feedback.** The public 5-minute window and
  `validate_submission.py` are the only signals you get before the deadline — so validate the exact
  file you are about to upload, not one you built earlier.

---

## Rules

The full rules are in **[RULES.md](RULES.md)**. The two that define the task rather than the
competition:

- **The labels must be produced by an algorithm.** Annotating the data by hand is not a submission.
- **You may not use the private ground truth** or anything derived from it, beyond the 5 minutes
  published here.
