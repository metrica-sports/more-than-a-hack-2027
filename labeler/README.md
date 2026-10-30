# ⚽ Barca Vision Labeler -- Tactical Annotation & Tracking Studio

**Barca Vision Labeler** is a high-performance, browser-based tactical video annotation and tracking software designed specifically for the **More Than A Hack 2027** ball possession challenge.

It allows you to review tracking frames at 25 fps, visualize team formations and player-ball physics on an interactive 2D pitch radar, receive instant AI pre-annotations, and rapidly label possession spells with professional video-editor hotkeys.

---

## 🚀 Quick Launch

### Option 1: One-Click Launch (Windows)
Double-click `run_labeler.bat` in either the workspace root or the `repo/` directory.

### Option 2: Command Line
```powershell
python labeler/run_labeler.py
```
This builds an $O(1)$ random-access index of all 199,319 tracking frames, boots the local Flask server on `http://127.0.0.1:5000`, and opens your default web browser automatically.

---

## 🌟 Core Features

### 1. Interactive 2D Tactical Pitch Radar
- **Hardware-Accelerated HTML5 Canvas**: Renders the 105.3m × 68.0m regulation football pitch.
- **FC Barcelona Players**: Blaugrana tokens (slots 1–23) with jersey numbers and dynamic velocity direction vectors.
- **Athletic Club Players**: Red/white tokens (slots 24–46) with jersey numbers and velocity vectors.
- **Smart Ball Tracker & Trajectory Trail**: Pulsing golden ball with a 12-frame trajectory trail and distance lines to the nearest home and away players.
- **Occlusion Handling**: Displays an amber warning banner when the ball coordinate is untracked (`NaN`) and visualizes team centroid shapes.

### 2. Multi-Track Timeline & Possession Ribbon
- **User Annotation Track**: Color-coded possession ribbon (Blue = Home `H`, Red = Away `A`, Amber = Dead Ball `D`).
- **Interactive Scrubber**: Click or drag to jump anywhere in the 199,319-frame match.
- **Multi-Speed Playback**: 25 fps real-time playback, 0.25x slow-mo, 0.5x, and 2.0x fast-forward.
- **Quick Jump Anchors**: One-click jump to 1st Half Kickoff (frame 11,185) or 2nd Half Kickoff (frame 113,741).

### 3. Rapid Spell / Range Labeling (10x Workflow)
In football, possessions last ~14 seconds (350 frames). Labeling frame-by-frame is unnecessary:
1. Scrub to where possession begins $\to$ Press **`I`** (Mark In-Point).
2. Scrub to where possession ends $\to$ Press **`O`** (Mark Out-Point).
3. Press **`H`**, **`A`**, or **`D`** $\to$ Instantly labels all frames in that range with clean continuous boundaries!

### 4. AI Pre-Annotation Assist (Active Learning)
- Built-in heuristic and temporal model estimates the most likely possession state in real time based on player proximity margin ($d_{away} - d_{home}$) and collective velocity decay.
- Shows live confidence percentage ($P(H), P(A), P(D)$).
- Press **`Enter`** to accept the AI suggestion for the active frame or marked range.

### 5. Seamless ML Pipeline Integration
- **`Load Reference GT`**: Loads the official 5-minute competition ground truth (`frames 11,185 .. 18,684`) into your workspace so you can inspect benchmark transitions.
- **`Export Ground Truth CSV`**: Writes official `frame,label` file to `data/annotated_possession_labels.csv` compatible with `evaluation/validate_submission.py`.
- **`Export ML Training Set`**: Compiles all labeled frames and automatically merges them with the 31 extracted collective features (`data/custom_training_dataset.csv`), ready for retraining `train_cnn_architectures.py`.

---

## ⌨️ Keyboard Shortcuts Cheat Sheet

| Key | Action | Description |
|---|---|---|
| **`Space`** | Play / Pause | Starts or stops 25 fps real-time playback |
| **`←` / `→`** | Step 1 Frame | Single-frame precision stepping backward / forward |
| **`Shift + ←` / `→`** | Step 1 Second | Jump 25 frames backward / forward |
| **`Ctrl + ←` / `→`** | Step 5 Seconds | Jump 125 frames backward / forward |
| **`H`** | Label **HOME** | Tag active frame or selection as Barcelona (`H`) |
| **`A`** | Label **AWAY** | Tag active frame or selection as Athletic Club (`A`) |
| **`D`** | Label **DEAD BALL** | Tag active frame or selection as Stoppage / Out of Play (`D`) |
| **`X` / `Delete`** | Clear Label | Clear label for active frame or selection |
| **`I` / `[`** | Mark In-Point | Sets start boundary for range labeling |
| **`O` / `]`** | Mark Out-Point | Sets end boundary for range labeling |
| **`R`** | Reset Range | Clears active In-Out selection |
| **`Enter`** | Accept AI | Confirm AI suggestion for current frame/range |

---

## 🔄 End-to-End Workflow: Expanding Your Training Data

1. **Launch Studio**: Double-click `run_labeler.bat`.
2. **Review & Label**:
   - Jump to frame 18,685 (right after the 5-minute public ground truth ends).
   - Use `I` and `O` to mark possession spells up to frame 33,684 (15 minutes).
   - Assign labels using `H`, `A`, and `D`.
3. **Export Training Data**:
   - Click **`Export ML Training Set`**.
   - Generates `data/custom_training_labels.csv` and merges with `data/custom_training_dataset.csv`.
4. **Retrain the Top Models**:
   ```powershell
   python starter/train_cnn_architectures.py
   ```
   The models (ResNet-1D and InceptionTime) will now train on 3x more data, pushing `spellF1` even higher!
