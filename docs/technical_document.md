# <Team name> — technical document

*More than a Hack 2027, frame-level ball possession. A few pages is plenty; delete the hints.*

## 1. Result
- Submission file: `<team_name>.csv`, 199,319 rows, passes `evaluation/validate_submission.py`.
- Public-window score (sanity check only, never used for tuning): macroF1 __ / spellF1 __.
- Possessions reported: __ (mean __ s).

## 2. Approach
What we built and why this rather than something else. Pipeline in one paragraph, then per stage:
1. Ball recovery (interpolation of gaps ≤ __ frames, confidence).
2. Control evidence per frame (features, how they combine).
3. Dead-ball detection.
4. Temporal model (Viterbi, switch costs) and why it beats frame-by-frame voting.

## 3. Assets used and dropped
| asset | used? | how / why not |
|---|---|---|
| tactical ATD | | |
| Smart Tagging | | |
| tactical homography | | |
| tactical video | | |
| broadcast ATD / video / homography | | |

## 4. Training data
Every labelled dataset used: source, who labelled it, how much. **None may come from the provided
game.** If nothing was trained, say so.

## 5. Possession rules implemented
Where we drew the line on: loose balls and deflections, aerial duels, challenges, restarts
(throw-ins, corners, goal kicks, kick-offs), second-half pre-kickoff frames (all `D`), how `X`
is used (inside a possession vs at a hand-over).

## 6. Missing and noisy tracking
How the 41% missing ball, duplicated or swapped slots, and the 46-slots-for-22-players issue are
handled.

## 7. Parameters
Each value in `possession/config.py`, and the football/other-game reason it was chosen. State
explicitly that none were tuned on the public 5 minutes.

## 8. Limitations
Where it fails and what we would fix with more time.

## 9. Reproduction
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-pipeline.txt
# data bundle in ./data (or export MTAH_DATA_DIR=...), see DATA.md
python run_pipeline.py --team <team_name>
```
Output: `submissions/<team_name>.csv` (deterministic; no randomness). Python __, numpy 1.26.1.

## 10. Team and LLM use
Members; which LLMs were used for code/writing and confirmation that no match data or video was
sent to a hosted LLM.
