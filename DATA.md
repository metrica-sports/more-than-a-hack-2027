# The data

One match: **FC Barcelona vs Athletic Club**, LaLiga EA Sports 2024-25, matchday 2.
Home is Barcelona (`H`), away is Athletic (`A`).

It was recorded twice, by two different cameras, and you get tracking for both.

| | **tactical** | **broadcast** |
|---|---|---|
| **role** | **the reference. Row *i* of your submission is frame *i* of this video** | optional extra input, **not synchronised** to tactical |
| video | 199,319 frames, 25 fps, 1920×1080 | its own timeline, with cuts and replays |
| tracking (ATD) | 199,319 records | 164,488 records |
| homography | 199,319 rows, **100%** usable | 164,488 rows, **65.8%** usable |
| Smart Tagging | 638 instances | — |
| track slots | 46 | 51 |
| ball visible | 56.9% of frames | 50.5% of frames |

Everything is fair game, in any combination. A system built only on the ball track is as
legitimate as one that reads the video. There is no intended solution.

> **If your submission has ~164k rows, you indexed the broadcast video.** The row-count check will
> reject it, which is exactly why that check exists.

---

## Getting it

The match assets are gated. They are released to participants who have enrolled and accepted the
terms of data use.

1. Enrol: **<TBD: enrolment form URL>**
2. Read and sign the terms and conditions: **<TBD: terms URL>**
3. You receive a download link.
4. Unpack it into `data/`, keeping the names it arrives with:

```
data/
├─ tactical/
│  ├─ BAR-ATH_tactical.mp4
│  ├─ BAR-ATH_tactical_atd.xml              tracking metadata
│  ├─ BAR-ATH_tactical_atd.txt              tracking, players and ball
│  ├─ BAR-ATH_tactical_homography.csv
│  └─ BAR-ATH_tactical_smart_tagging.xml
└─ broadcast/
   ├─ BAR-ATH_broadcast.mp4
   ├─ BAR-ATH_broadcast_atd.xml
   ├─ BAR-ATH_broadcast_atd.txt
   └─ BAR-ATH_broadcast_homography.csv
```

Nothing in this repository hard-codes a filename — `games.py` builds them all, and
`MTAH_DATA_DIR` moves the whole bundle elsewhere without any other change:

```python
from games import asset_paths
asset_paths()["atd_raw"]                    # tactical, the default
asset_paths(source="broadcast")["video"]
```

Everything in `data/` is gitignored. Do not commit it and do not redistribute it — the terms you
signed cover what you may and may not do with it.

**The evaluation needs none of this.** It reads CSVs only, so the scorer runs on a fresh clone
before you have any data at all.

---

## Automatic Tracking Data (ATD)

**What it is.** The output of Metrica's computer-vision pipeline: the position and speed of every
object it can see on the pitch, once per video frame, **including the ball**. It is generated
automatically from the video — not corrected, not manually reviewed.

Metrica's FIFA/EPTS export, a metadata XML plus a raw text file. `kloppy` reads the pair directly:

```python
from kloppy import metrica
from games import asset_paths

paths = asset_paths()
dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])
```

`dataset.to_df()` gives one row per frame: `frame_id`, `ball_x`, `ball_y`, then `1_x`, `1_y`, `1_s`
for each track slot — position, and speed in m/s. At 25 fps, 25 rows is one second.

**Coordinates are normalised to 0–1**, not metres. Multiply by the pitch to get metres:

```python
dataset.metadata.pitch_dimensions.pitch_length   # 105
```

The metadata reports 105 × 68 m; `games.py` carries 105.3 × 68.0, measured from the corrected
tracking. The 30 cm matters only if you are computing distances in metres.

### What it does not contain

**No player identities.** There are **46 track slots** for 22 players on the pitch. They are named
`Track 1`…`Track 46`, and their `ShirtNumber` is the slot index — **not a shirt number, and not a
person**. Slots appear, vanish and swap identity as the tracker loses and re-acquires people. You
get positions and team membership, nothing more, so any analysis you build has to be collective
rather than per-player.

Expect around 22 tracked objects during play — the mean over graded frames is 20.5, and it
occasionally exceeds 22 when the tracker duplicates someone.

### The ball is missing 43% of all frames, and 41% of the graded ones

The most important single fact about the dataset. It is present on **56.9%** of all frames
(113,488 / 199,319) and **59.3%** of graded ones (94,214 / 158,857).

The ball is the most informative signal you have and the least reliable one — occlusions, crowded
boxes and balls in the air all lose it, which means it goes missing disproportionately on the
frames that matter. Any approach assuming a ball position exists on every frame will meet this on
day two, so design for it on day one. Missing observations are `NaN`, for players and ball alike.

`starter/02_tracking_atd.ipynb` walks through loading and plotting;
`starter/03_first_submission.ipynb` turns the ATD into a scored submission.

---

## Smart Tagging

Metrica's **automatic** tactical annotation of the tactical video: 638 phases of play and set
pieces, each an interval with a code and a set of tags. A SportsCode XML file.

```python
from kloppy import sportscode
from games import asset_paths
events = sportscode.load(asset_paths()["smart_tagging"]).to_df()
```

You get `timestamp`, `end_timestamp` (pandas timedeltas), `code`, and one column per tag group.
**Tag columns are title case** — the column is `Team`, not `team`. Codes are upper case
(`SET PIECES`) and tag values are title case (`Kick Off (Start)`), so match case-insensitively
unless you enjoy silent empty results.

`starter/load_smart_tagging.py` does the same with the standard library, no kloppy needed.

| code | n | | code | n |
|---|---|---|---|---|
| `PLAYERS IN THE BOX` | 97 | | `CREATING CHANCES` | 30 |
| `BALL IN FINAL THIRD` | 93 | | `DEFENDING IN DEFENSIVE THIRD` | 30 |
| `BALL IN THE BOX` | 72 | | `LONG BALL` | 29 |
| `PROGRESSION` | 66 | | `ATTACKING TRANSITION` | 24 |
| `DEFENDING IN MIDDLE THIRD` | 66 | | `DEFENSIVE TRANSITION` | 24 |
| `SET PIECES` | 59 | | `BUILD UP` | 22 |
| | | | `DEFENDING IN ATTACKING THIRD` | 22 |
| | | | `GOALS` | 4 |

| tag group | values |
|---|---|
| `Team` | `Barcelona` (314) · `Atletic Club` (288) · **`N/A` (36)** |
| `Half` | `1st Half` · `2nd Half` — strings, not numbers |
| `Type` | `Throw-in` (26) · `Corner Kick` (12) · `Goal Kick` (10) · `Free Kick` (5) · `Kick Off (After Goal)` (4) · `Kick Off (Start)` (2) |
| `Side` | `Central` · `Left` · `Right` |
| `Direction of ball entry` | `Horizontal` · `Vertical` · `Diagonal` |
| `Max Players in the box` | `1` … `6`, and `7+` — a **string**, not an integer |

Note the club is spelled **`Atletic Club`**, with one `h`, in both the ATD metadata and this file.
Map club names to sides once, at the edge of your code.

### Time base — verified

`<start>` and `<end>` are **seconds from the first frame of the tactical video**, so
`round(seconds * 25)` is a video frame and there is nothing else to correct for. Checked against
both kickoffs; re-run it any time with `python starter/load_smart_tagging.py`.

### Why it is worth using, and what it is not

Every phase carries a team tag and every possession has a team; every set piece is a restart and
every restart ends a dead-ball stretch. That is a strong prior, for free, on both halves of the
label you are producing. But:

- **It is inferred, not ground truth.** Metrica generates it automatically from the same video and
  tracking you have, without manual review, so it contains false positives and false negatives like
  any automatic output. Treat a tag as evidence, not as an answer.
- **`Team` is not always a team.** 36 instances are tagged `N/A`, including **every throw-in** —
  exactly the restarts where knowing the side would be worth the most. Handle it explicitly.
- **The intervals are phases, not possessions.** A Build Up can span one possession, stop short of
  one, or cover two. Copying an interval across as a label imports its errors wholesale.

---

## Homography

**What it is.** A 3×3 matrix, one per video frame, that maps a position on the pitch to the pixel
that shows it. It is what connects the tracking to the video: project the players onto a frame to
see what your model saw, or take a pixel back to a pitch position.

A CSV with one row per video frame, `frame` counting from 0 — for the tactical recording that is
the same index as a submission row.

```
frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid
11185,1822.663,-511.2005,44.48168,1.698323,169.2971,255.7303,-0.01907188,-0.5269586,1,0.2910854,1
```

Columns 1–9 are the matrix in row-major order, so `row[1:10].reshape(3, 3)` is all it takes.

**It maps normalised pitch coordinates (0–1) to image pixels** — the same 0–1 convention the ATD
uses, so a tracking position goes straight in with no conversion:

```python
from starter.load_homography import load, project
H, reprojection_error, valid = load()
x, y = project(H[frame], tracking_x, tracking_y)     # -> pixels
```

**Check `valid` before using a matrix.** Where it is 0 the nine matrix columns are `NaN`, because
no pitch transform exists for that frame at all. The tactical file is valid everywhere. The
broadcast file is valid on 65.8% of frames — the rest are replays, close-ups and crowd shots, in
122 stretches, the longest over a minute and a half.

`reprojection_error` is how far the fitted matrix misses by, so **lower is better** and there is no
upper bound — the broadcast file reaches 304. On the tactical file the median is 0.24 during play
but about 1.5 across halftime, when the camera is not pointed at the pitch.

The two columns answer different questions: `valid` says whether a matrix exists at all,
`reprojection_error` says how much to trust the one you have.

---

## The broadcast recording

The TV feed: what a human analyst actually watches. It has its own ATD and its own homography, in
the same formats as above.

It is **not frame-aligned to the tactical video**, and no mapping table is provided. It has cuts,
replays and zooms, its own frame count, and a third of its frames have no usable homography.
Mapping it back to tactical frames is your problem if you choose to use it.

```python
asset_paths(source="broadcast")
```

---

## Frame alignment

Everything for a given recording is on that recording's timeline. There is no offset to discover.

| source | index | to get a submission row |
|---|---|---|
| **submission** | video frame `0 … 199318` | — |
| **ATD** | `frame_id` `1 … 199319` | `video_frame = frame_id - 1` |
| **Smart Tagging** | seconds from the start of the video | `video_frame = round(seconds * 25)` |
| **homography** | `frame` `0 … 199318` | already a submission row |

### Which frames are graded

The two halves, in video frames. These are published because they are useful and because you could
read them off the video anyway — they are boundaries, not answers, and they reveal no label:

| | first frame | last frame | frames | graded? |
|---|---|---|---|---|
| before the match | 0 | 11,184 | 11,185 | no |
| **1st half** | **11,185** | **90,651** | 79,467 | **yes** |
| halftime | 90,652 | 110,807 | 20,156 | no |
| **2nd half** | **110,808** | **190,197** | 79,390 | **yes** |
| after the final whistle | 190,198 | 199,318 | 9,121 | no |

158,857 graded frames, 40,462 ungraded. Whatever you write on an ungraded frame is ignored, so `X`
there costs nothing.

```python
from games import half_video_blocks
half_video_blocks("BAR-ATH")      # [(1, 11185, 90651), (2, 110808, 190197)]
```

---

## The public ground truth

`evaluation/gt/BAR-ATH_possession_gt_public.csv` — in this repository, no download needed. A
full-length file, 199,319 rows exactly like a submission, carrying real labels only for the **first
5 minutes of play**: video frames **11185 … 18684**, `H` 3,454 · `A` 2,577 · `D` 1,469, 17
possessions of mean 14.2 s. Everything outside is `X` and is never scored, so your full-match
submission validates against it unchanged. `..._public_meta.json` has the same numbers in
machine-readable form.

It is for checking that your pipeline is right, **not for ranking** — see
[EVALUATION.md](EVALUATION.md). The final ranking uses the full-match ground truth (158,857 graded
frames), which stays private.
