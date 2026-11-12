# Ball possession from numeric data only — technical document

Team submission for More Than A Hack 2027 (BAR-ATH). The method uses only the automatic tracking
(ATD) and the Smart Tagging of the tactical video. It has no video model and no human labels on the
provided game, and every parameter is calibrated on other games.

## 1. Approach

Each of the 199,319 tactical-video frames gets one of H / A / D / X. The ranking metric (spellF1)
scores **possession spells as segments**, so the output has to be temporally coherent: a per-frame
classifier that flickers between teams scores near zero (`baseline_raw`: spellF1 0.04) even when
most of its frames are right.

The pipeline therefore has two layers:

1. **Per-frame evidence** for H, A and D as log-likelihood scores (`possession/emissions.py`).
   Each frame adds a small amount of evidence and no single frame decides anything.
2. **A minimum-duration HMM** decoded with Viterbi, independently for each half
   (`possession/decode.py`). Each state H / A / D is expanded into a chain of `min_len`
   sub-states, so a state can only be left after it has lasted `min_len` frames: a 3-frame
   possession is impossible by construction. Leaving a state costs a fixed log-penalty
   (`switch`). Frames outside the two graded halves are written as X. Inside a half we never
   abstain, because X on a graded frame is always a miss.

We chose an explicit HMM over an end-to-end sequence model (e.g. the TCN of Borghesi et al. 2023)
because labelled training data from other games is scarce (two open games) and the tracking
domain differs between providers. A model with a dozen physically meaningful parameters,
calibrated directly on the ranking metric, is less exposed to overfitting and is easy to inspect.

### 1.1 Per-frame evidence

Features from the tracking (`possession/features.py`, metres and m/s):

- distance from the ball to the nearest player of each team;
- ball speed (centred difference over ±2 frames, then a 5-frame median to absorb tracker jumps);
- how far the ball is outside the pitch;
- collective player speed (mean over visible players, 2 s rolling mean; values above 12 m/s
  are discarded as identity switches);
- ball stillness over 1 s.

Evidence terms:

- **Team control**, after Link & Hoernig (2017). Control decays exponentially beyond
  `control_radius` (scale `control_decay`) and is down-weighted when the ball moves faster than
  `flight_speed` (a ball in flight is controlled by nobody). The evidence is the difference
  between home and away control, weight `k_team`, and is used only where the ball is tracked.
- **Dead ball**: the ball beyond the line by more than `outside_margin` (`k_outside`), slow
  collective player speed (a tanh around `slow_center`, `k_slow`), a still ball with slow
  players (`k_still`), and a constant prior against D (`d_bias`).
- **Dead-ball classifier** (section 1.3), weight `k_dead`.
- **Smart Tagging prior** (section 1.4).

### 1.2 Loose balls, challenges and restarts

These are not hand-coded rules; they follow from the evidence and the HMM:

- **Loose balls and passes**: a ball in flight gives weak or no team evidence, so the decoder
  keeps the current team until the evidence for the other team outweighs the `switch` penalty.
  A pass between team-mates therefore never ends a spell.
- **Challenges and short contested phases**: a change of team only happens once the new team
  has stayed in control for at least `min_pos` frames, so 50/50 duels shorter than that are
  absorbed by the surrounding spell.
- **Restarts**: a stoppage is a D run of at least `min_dead` frames. The spell after it starts
  wherever the team evidence (and, for set pieces, the Smart Tagging) points once play resumes.

### 1.3 Dead-ball classifier on every frame (`possession/deadball.py`)

The automatic tracking often still sees the ball during a stoppage, for example a player holding
it for a throw-in. Nearest-player evidence then reads as a clear H or A. We built a classifier of
P(dead ball) on **every** frame from the restart patterns described by Vidal-Codina et al. (2022).
It uses 25 multi-scale features:

- the ball still at a restart spot: touchline (throw-in), corner flag (corner), 6-yard box line
  (goal kick), or centre spot with each team in its own half (kick-off);
- ball stillness over 1 s and 3 s, and the maximum ball speed over the past and next 2 s;
- player speed now, over the past 5 s and the next 3 s, the 5 s minimum, and the 90th percentile.
  Together these capture the "slow down → still → burst" shape of a restart. All player speeds
  are normalised by the game median to absorb tracker differences;
- nearest-player distance, horizontal spread of the players, players within 10 m of the ball;
- time since the ball was last tracked and until it is tracked again (gap context).

The model is a scikit-learn `HistGradientBoostingClassifier` (300 iterations, 31 leaves, at least
200 samples per leaf, L2 1.0, every 4th frame). Its log-odds, centred on the training prior and
clipped to ±4, enters the D score with weight `k_dead`. It is saved as `deadball_model.pkl`.

A first, narrower version, a logistic-regression classifier only for frames without the ball
(`possession/gaps.py`), is kept in the code but switched off (`k_gap = 0`). On the calibration
games it added nothing beyond the collective player speed, and the dead-ball classifier already
sees the gap context.

### 1.4 Smart Tagging

- **Team tags**: a weak prior (`k_tag_team`) towards the team named by a phase. We decide
  algorithmically whether a code names the team in possession or the opponent. For each code we
  measure agreement with the nearest-player team on frames where someone is clearly on the ball
  (within 1.5 m). No ground truth is involved.
- Team tags of `SET PIECES` and `GOALS` are not used as phase priors: they refer to the restart,
  not to the whole window, and are handled by the anchors below.

#### Smart Tagging set-piece anchors (`possession/setpieces.py`)

Every `SET PIECES` instance says that a restart happens in (or near) its few-second window, and
its `Type` says which one. We use each as an anchor:

1. **Restart frame from the tracking.** Within the tag window ±3 s we look for the frame where the
   ball goes from still (mean speed over the previous second) to moving (peak speed over the next
   second above 3 m/s), with a bonus when the ball is at the spot the Type implies (touchline for a
   throw-in, corner flag, 6-yard box for a goal kick, centre spot for a kick-off). If the ball is not
   tracked, the frame where the players speed up the most; otherwise the window centre. On the
   Metrica games with synthetic tags the median error is 0.6–0.8 s (65% within 1 s, ~80% within 2 s).
2. **Dead ball before it.** D evidence (`k_sp_dead`) on a ramp ending at the restart, over a
   type-dependent duration (median dead-ball time before each restart type in the Metrica games:
   throw-in 7 s, free kick 12 s, goal kick 20 s, corner 28 s, kick-off after a goal 55 s), scaled by
   `sp_back_scale`.
3. **Possession of the taker after it.** For 3 s after the restart, evidence for the taker's team
   and against D (`k_sp_team`). The taker is the tagged team; whether `Team` names the taker or the
   opponent is decided by agreement with the nearest player right after the detected restarts (no
   ground truth). Throw-ins are tagged `N/A`, so there the taker is the nearest player.
4. **Goals.** From the end of a `GOALS` window to the next kick-off restart: D.

**Calibrating the tag weights without tags.** Metrica has no Smart Tagging, so we synthesise it from
the Metrica events (`metrica.synthetic_tags`): a 3–7 s window placed randomly around each restart
(1–4 s before it), the Type, the taker's team (`N/A` for throw-ins), and — the important part —
many misses. Keep rates per type are the restarts in the real tag file over typical counts per match
(throw-in 0.55, corner 0.9, goal kick 0.65, free kick 0.25, kick-offs 1.0), plus two false positives
per half at random live moments. The weights `k_sp_dead`, `k_sp_team`, `sp_back_scale` are then
searched with everything else.

On Metrica (100 paired configurations, same parameters with and without anchors) the anchors improve
the mean spellF1 by +0.018 and help in 100/100 configurations, but the best worse-game spellF1 moves
only from 0.407 to 0.412 — within noise. On Metrica the dead-ball classifier already finds 84% of
the stoppages; the anchors are aimed at the automatic tracking, where it finds only a third of them.

## 2. Assets used and dropped

| asset | used? | how / why not |
|---|---|---|
| Tactical ATD (tracking) | **yes** | ball and player positions → all features |
| Tactical Smart Tagging | **yes** | team and set-piece prior (section 1.4) |
| Tactical / broadcast video | no | team decision: numeric data only, within the time and compute we had |
| Homography (tactical, broadcast) | no | positions in the ATD are already in pitch coordinates |
| Broadcast ATD | no | covers a different video, and the submission is indexed on the tactical video |
| LLMs | no | not used on the match data; the T&C forbid uploading it to hosted LLMs |

## 3. Training data

All labelled data comes from **other games**:

- **Metrica Sports open sample data**, Sample Games 1 and 2: tracking and event data, public on
  GitHub (https://github.com/metrica-sports/sample-data). The data is downloaded automatically by
  `possession/metrica.py`.
- **Labels**: derived **algorithmically** from Metrica's published event data
  (`metrica.labels_from_events`). We did not label anything by hand.
  - Control events (PASS, SHOT, RECOVERY, SET PIECE, BALL LOST) set the team from their start
    frame.
  - Dead-ball triggers (BALL OUT, FAULT RECEIVED, SHOT out or goal, BALL LOST by offside or end of
    half) set D from the trigger (for ball-out and shots, from the end of the event) until the
    next restart (SET PIECE or goal kick).
  - Frames before the first control event and after the last event of each period are X.
- **Nothing from the provided game** is used for training, calibration or validation. The public
  ground-truth window is used only as an end-of-pipeline sanity check. No parameter is tuned on
  it.

### 3.1 Making Metrica look like the automatic tracking ("atd_like" protocol)

In the Metrica samples the ball is missing almost only during stoppages (92–100% of D frames
against 9–15% of live frames). In the challenge ATD a missing ball says little about a stoppage.
Calibrating on raw Metrica would teach the model a shortcut that does not exist in the target data,
so `metrica.atd_like_ball`:

1. fills the native Metrica ball gaps by linear interpolation within each period;
2. cuts long random gaps (log-normal lengths, median 268 frames ≈ 11 s) until 41% of frames have
   no ball, mimicking the few, long gaps of the ATD.

With this protocol the best configuration without the dead-ball classifier reaches spellF1 ≈ 0.37
on the worse calibration game, against the ≈ 0.44 seen on raw Metrica. The difference was the
shortcut.

## 4. Calibration

- **Objective**: the official spellF1 (`evaluation/possession_metrics.py`) on the two Metrica
  games, atd_like ball, ball gaps drawn with a fixed seed.
- **Search**: seeded random search over `k_team`, `control_decay`, `flight_speed`, `slow_center`,
  `k_slow`, `k_still`, `d_bias`, `k_dead`, `k_sp_dead`, `k_sp_team`, `sp_back_scale` and the decoder (`switch` 2–15, `min_pos` 5–75 frames,
  `min_dead` 10–150 frames). The search runs in parallel with checkpoint and resume
  (`calibrate.parallel_search`).
- **Leave-one-game-out**: during calibration, the dead-ball classifier used on each game is trained
  on the other game only. The final classifier for the challenge game is trained on both games.
- **Selection**: `best_of` keeps the configuration whose **worse** game scores highest, to avoid
  rewarding a single lucky game.
- Smart Tagging: the set-piece anchor weights are calibrated with synthetic tags (1.4); the phase
  team-tag weight `k_tag_team` stays hand-set (phases cannot be synthesised faithfully).

### 4.1 Results on the calibration games (Metrica, atd_like, leave-one-game-out)

| | without dead-ball classifier | with |
|---|---|---|
| spellF1, worse game (best config of 100) | 0.374 | **0.413** |
| configs improved (paired, same params) | – | 100 / 100 |
| macroF1 (two games) | 0.72 / 0.70 | 0.76 / 0.75 |
| D recall (two games) | 0.78 / 0.68 | 0.84 / 0.84 |
| dead-ball AUC, out of game | 0.87 / 0.80 (player speed alone) | 0.925 / 0.914 |

A logistic regression on the same features reaches 0.92 / 0.91 AUC and 0.397 worse-game spellF1.
We kept gradient boosting.

## 5. Limitations

- **Domain shift, Metrica → ATD.** On the public window the dead-ball recall is 0.33 (precision
  0.89), against 0.84 on Metrica. Two stoppages in three are still read as play, which merges
  possessions: 14 predicted spells against 17, mean duration 19.9 s against 14.2 s. Before the
  dead-ball classifier the recall was 0.08–0.11 and only 9 spells were predicted.
- **Team errors.** Team accuracy on live frames is about 0.83, and some whole spells are given to
  the wrong team. Nearest-player control alone agrees with the truth about 76% of the time.
- **Boundaries.** Few transitions land within ±1 s of the true one. spellF1 drops quickly with the
  IoU threshold (0.52 at 0.3 → 0.13 at 0.7 on the public window).
- **Two calibration games only.** The public window has 17 possessions, so its bootstrap interval
  is very wide (spellF1 5–95%: 0.18–0.50). Differences below the scorer's noise band are not
  meaningful.

With more time: a team classifier built like the dead-ball one, more open games for calibration
(other providers, converted to the same format).

## 6. Reproduction

```bash
pip install -r requirements.txt -r solution/requirements.txt      # exact versions, see below
# data bundle unpacked in data/ (or MTAH_DATA_DIR)
python solution/run.py --out evaluation/results/<team_name>.csv     # reads solution/params.json
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

- `solution/params.json` (calibrated parameters) and `solution/deadball_model.pkl` (dead-ball
  classifier) are committed with the final submission. `run.py` regenerates the same file from
  them and stops if either is missing.
- `solution/requirements.txt` pins numpy 2.2.6, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1,
  matplotlib 3.10.3. The pickle must be loaded with the scikit-learn version that wrote it, which
  is stored in the file.
- To redo the calibration: notebooks `solution/notebooks/02_calibrazione_metrica.ipynb` (writes
  both files) and `03_pipeline_e_submission.ipynb` (writes and checks the submission), on Google
  Colab or locally. Timing: ATD parsing about 1 min, pipeline about 5 s, calibration a few seconds
  per configuration per core.

## References

- Vidal-Codina, F. et al. (2022). *Automatic event detection in
  football using tracking data.* https://arxiv.org/abs/2202.00804
- Link, D., Hoernig, M. (2017). *Individual ball possession in soccer.* PLOS ONE.
  https://doi.org/10.1371/journal.pone.0179953
- Borghesi, M. et al. (2023). *Using temporal convolutional networks to estimate ball possession
  in soccer games.* Expert Systems with Applications 223.
  https://www.sciencedirect.com/science/article/pii/S0957417423002816
- Kim, H. et al. (2023). *Ball Radar.* https://arxiv.org/abs/2306.08206
- Capellera, G. et al. (2024). *TranSPORTmer.* https://arxiv.org/abs/2410.17785
- Metrica Sports sample data: https://github.com/metrica-sports/sample-data
