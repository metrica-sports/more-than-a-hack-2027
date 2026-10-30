"""Barca Vision Labeler: High-Performance Tactical Annotation & Tracking Studio.

Flask backend serving real-time 25 fps tracking data, byte-offset frame indexing,
AI model pre-annotations, spell/interval labeling, and ML dataset export.
"""

import os
import sys
import json
import math
import time
from typing import Dict, List, Optional, Tuple
from flask import Flask, jsonify, request, render_template, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from games import GAMES, GAME  # noqa: E402

app = Flask(__name__, template_folder=os.path.join(HERE, "templates"))

# Path configurations
PITCH_LEN, PITCH_WID = GAMES[GAME]["pitch"]  # 105.3m, 68.0m
FPS = 25
TOTAL_FRAMES = GAMES[GAME]["n_video_frames"]  # 199319
RAW_ATD_PATH = os.path.join(REPO_ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt")
GT_PUBLIC_PATH = os.path.join(REPO_ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
SESSION_FILE = os.path.join(REPO_ROOT, "data", "labeler_session.json")
EXPORT_CSV_PATH = os.path.join(REPO_ROOT, "data", "annotated_possession_labels.csv")

# Global state
BYTE_OFFSETS: List[int] = []
REFERENCE_GT: Dict[int, str] = {}
USER_LABELS: Dict[int, str] = {}  # {video_frame: 'H'|'A'|'D'}


def init_byte_offsets():
    """Build an in-memory index of byte offsets for O(1) random frame access."""
    global BYTE_OFFSETS
    if BYTE_OFFSETS:
        return
    t0 = time.time()
    offsets = []
    if os.path.exists(RAW_ATD_PATH):
        with open(RAW_ATD_PATH, "rb") as f:
            offset = 0
            for line in f:
                offsets.append(offset)
                offset += len(line)
        BYTE_OFFSETS = offsets
        print(f"[Labeler] Indexed {len(BYTE_OFFSETS)} frames in {time.time() - t0:.2f}s")
    else:
        print(f"[Labeler] Warning: Raw tracking file not found at {RAW_ATD_PATH}")


def init_reference_gt():
    """Load the official public ground truth for comparison and reference."""
    global REFERENCE_GT
    if os.path.exists(GT_PUBLIC_PATH):
        with open(GT_PUBLIC_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("frame"):
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    vf = int(parts[0])
                    lbl = parts[1].strip()
                    if lbl in ("H", "A", "D"):
                        REFERENCE_GT[vf] = lbl
        print(f"[Labeler] Loaded {len(REFERENCE_GT)} reference GT frames from {GT_PUBLIC_PATH}")


def load_saved_session():
    """Load previously saved user labeling session if available."""
    global USER_LABELS
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                USER_LABELS = {int(k): v for k, v in data.items() if v in ("H", "A", "D")}
            print(f"[Labeler] Restored session with {len(USER_LABELS)} labeled frames")
        except Exception as e:
            print(f"[Labeler] Could not load session file: {e}")


def save_session():
    """Persist user annotations to session JSON."""
    os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(USER_LABELS, f)


def parse_raw_line(line_str: str) -> Tuple[int, List[dict], Optional[dict]]:
    """Parse raw ATD line string into structured players and ball dict."""
    parts = line_str.strip().split(":")
    frame_id = int(parts[0])
    players_raw = parts[1].split(";")
    players = []

    for slot_idx, p in enumerate(players_raw):
        if not p or p.startswith("NaN"):
            continue
        vals = p.split(",")
        try:
            x_norm = float(vals[0])
            y_norm = float(vals[1])
            s = float(vals[2]) if len(vals) > 2 else 0.0
            if math.isnan(s):
                s = 0.0
            team = "H" if slot_idx < 23 else "A"
            players.append({
                "slot": slot_idx + 1,
                "team": team,
                "x": x_norm,
                "y": y_norm,
                "s": round(s, 2)
            })
        except (ValueError, IndexError):
            continue

    ball = None
    if len(parts) > 2 and not parts[2].startswith("NaN"):
        bvals = parts[2].split(",")
        try:
            bx = float(bvals[0])
            by = float(bvals[1])
            bz = float(bvals[2]) if len(bvals) > 2 else 0.0
            bs = float(bvals[3]) if len(bvals) > 3 else 0.0
            if not (math.isnan(bx) or math.isnan(by)):
                ball = {
                    "x": bx,
                    "y": by,
                    "z": 0.0 if math.isnan(bz) else round(bz, 2),
                    "s": 0.0 if math.isnan(bs) else round(bs, 2)
                }
        except (ValueError, IndexError):
            ball = None

    return frame_id, players, ball


def compute_frame_telemetry(players: List[dict], ball: Optional[dict]) -> dict:
    """Compute physical metrics, ball proximity, and AI possession heuristic."""
    h_players = [p for p in players if p["team"] == "H"]
    a_players = [p for p in players if p["team"] == "A"]

    h_cx = sum(p["x"] for p in h_players) / len(h_players) if h_players else 0.5
    h_cy = sum(p["y"] for p in h_players) / len(h_players) if h_players else 0.5
    a_cx = sum(p["x"] for p in a_players) / len(a_players) if a_players else 0.5
    a_cy = sum(p["y"] for p in a_players) / len(a_players) if a_players else 0.5

    h_speed = sum(p["s"] for p in h_players) / len(h_players) if h_players else 0.0
    a_speed = sum(p["s"] for p in a_players) / len(a_players) if a_players else 0.0
    overall_speed = sum(p["s"] for p in players) / len(players) if players else 0.0

    nearest_h = None
    nearest_a = None
    min_dh = 999.0
    min_da = 999.0

    if ball is not None:
        bx_m = ball["x"] * PITCH_LEN
        by_m = ball["y"] * PITCH_WID

        for p in players:
            px_m = p["x"] * PITCH_LEN
            py_m = p["y"] * PITCH_WID
            dist = math.hypot(px_m - bx_m, py_m - by_m)
            p["dist_ball"] = round(dist, 2)

            if p["team"] == "H" and dist < min_dh:
                min_dh = dist
                nearest_h = {"slot": p["slot"], "dist": round(dist, 2)}
            elif p["team"] == "A" and dist < min_da:
                min_da = dist
                nearest_a = {"slot": p["slot"], "dist": round(dist, 2)}

        margin = round(min_da - min_dh, 2)
    else:
        for p in players:
            p["dist_ball"] = None
        margin = 0.0

    # Heuristic AI Suggestion
    ai_label = "D"
    ai_conf = 0.50
    probs = {"H": 0.33, "A": 0.33, "D": 0.34}

    # If ball is present and someone is close or clear margin exists -> Live possession
    ball_is_active = (ball is not None) and (min_dh < 4.0 or min_da < 4.0 or abs(margin) > 2.5)

    if overall_speed < 0.90 and not ball_is_active:
        # High likelihood of stoppage / dead ball
        ai_label = "D"
        ai_conf = 0.85
        probs = {"H": 0.07, "A": 0.08, "D": 0.85}
    elif ball is not None:
        if margin > 1.2:
            ai_label = "H"
            ai_conf = min(0.95, 0.60 + margin * 0.06)
            probs = {"H": round(ai_conf, 2), "A": round((1 - ai_conf) * 0.7, 2), "D": round((1 - ai_conf) * 0.3, 2)}
        elif margin < -1.2:
            ai_label = "A"
            ai_conf = min(0.95, 0.60 + abs(margin) * 0.06)
            probs = {"A": round(ai_conf, 2), "H": round((1 - ai_conf) * 0.7, 2), "D": round((1 - ai_conf) * 0.3, 2)}
        else:
            # Contested ball
            if margin >= 0:
                ai_label = "H"
                ai_conf = 0.55
                probs = {"H": 0.55, "A": 0.35, "D": 0.10}
            else:
                ai_label = "A"
                ai_conf = 0.55
                probs = {"A": 0.55, "H": 0.35, "D": 0.10}
    else:
        # Ball occluded: check speeds and centroid
        if overall_speed < 1.0:
            ai_label = "D"
            ai_conf = 0.75
            probs = {"H": 0.12, "A": 0.13, "D": 0.75}
        elif h_cx > 0.55:
            ai_label = "H"
            ai_conf = 0.65
            probs = {"H": 0.65, "A": 0.25, "D": 0.10}
        elif a_cx < 0.45:
            ai_label = "A"
            ai_conf = 0.65
            probs = {"A": 0.65, "H": 0.25, "D": 0.10}

    return {
        "nearest_home": nearest_h,
        "nearest_away": nearest_a,
        "margin": margin,
        "home_centroid": {"x": round(h_cx, 3), "y": round(h_cy, 3)},
        "away_centroid": {"x": round(a_cx, 3), "y": round(a_cy, 3)},
        "home_speed": round(h_speed, 2),
        "away_speed": round(a_speed, 2),
        "overall_speed": round(overall_speed, 2),
        "ai_suggestion": {
            "label": ai_label,
            "confidence": round(ai_conf, 2),
            "probs": probs
        }
    }


def get_single_frame(video_frame: int) -> Optional[dict]:
    """Retrieve full data for a single video frame."""
    if video_frame < 0 or video_frame >= len(BYTE_OFFSETS):
        return None

    offset = BYTE_OFFSETS[video_frame]
    with open(RAW_ATD_PATH, "r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        line = f.readline()

    frame_id, players, ball = parse_raw_line(line)
    telemetry = compute_frame_telemetry(players, ball)

    # Format timestamp MM:SS.ff
    seconds_total = video_frame / float(FPS)
    minutes = int(seconds_total // 60)
    seconds = int(seconds_total % 60)
    subframes = int((seconds_total - int(seconds_total)) * 100)
    timestamp = f"{minutes:02d}:{seconds:02d}.{subframes:02d}"

    half = 1 if video_frame < 100000 else 2

    return {
        "video_frame": video_frame,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "half": half,
        "players": players,
        "ball": ball,
        "telemetry": telemetry,
        "label": USER_LABELS.get(video_frame),
        "gt_label": REFERENCE_GT.get(video_frame)
    }


# ===================== FLASK ROUTES =====================

@app.route("/")
def index():
    """Serve the primary tactical labeling interface."""
    return render_template("index.html")


@app.route("/api/config")
def api_config():
    """Return match dimensions, frame bounds, and initial state."""
    return jsonify({
        "game": GAME,
        "total_frames": TOTAL_FRAMES,
        "fps": FPS,
        "pitch": [PITCH_LEN, PITCH_WID],
        "default_kickoff": 11185,
        "user_labels_count": len(USER_LABELS),
        "gt_count": len(REFERENCE_GT)
    })


@app.route("/api/frame/<int:video_frame>")
def api_frame(video_frame):
    """Retrieve detailed data for a specific frame."""
    frame_data = get_single_frame(video_frame)
    if frame_data is None:
        return jsonify({"error": "Frame out of bounds"}), 404
    return jsonify(frame_data)


@app.route("/api/chunk")
def api_chunk():
    """Return a contiguous chunk of frames for smooth frontend playback."""
    start = int(request.args.get("start", 11185))
    length = int(request.args.get("length", 100))
    length = min(max(length, 1), 300)  # Bound chunk between 1 and 300 frames

    frames = []
    end = min(start + length, len(BYTE_OFFSETS))
    for vf in range(start, end):
        fd = get_single_frame(vf)
        if fd:
            frames.append(fd)

    return jsonify({
        "start": start,
        "end": end,
        "count": len(frames),
        "frames": frames
    })


@app.route("/api/label", methods=["POST"])
def api_label():
    """Tag a single frame or interval with H, A, D, or clear."""
    data = request.get_json(force=True)
    frame = data.get("frame")
    label = data.get("label")  # 'H', 'A', 'D', or None/'X' to clear

    if frame is None or not (0 <= frame < TOTAL_FRAMES):
        return jsonify({"error": "Invalid frame"}), 400

    if label in ("H", "A", "D"):
        USER_LABELS[frame] = label
    else:
        USER_LABELS.pop(frame, None)

    save_session()
    return jsonify({"success": True, "frame": frame, "label": USER_LABELS.get(frame)})


@app.route("/api/label_range", methods=["POST"])
def api_label_range():
    """Tag a range of frames [start, end] with H, A, D, or clear."""
    data = request.get_json(force=True)
    start = int(data.get("start", 0))
    end = int(data.get("end", 0))
    label = data.get("label")

    if start > end:
        start, end = end, start

    start = max(0, start)
    end = min(TOTAL_FRAMES - 1, end)

    count = 0
    for vf in range(start, end + 1):
        if label in ("H", "A", "D"):
            USER_LABELS[vf] = label
        else:
            USER_LABELS.pop(vf, None)
        count += 1

    save_session()
    return jsonify({"success": True, "start": start, "end": end, "updated_frames": count})


@app.route("/api/labels_overview")
def api_labels_overview():
    """Return compact run-length encoded spells and label statistics."""
    if not USER_LABELS:
        return jsonify({
            "spells": [],
            "stats": {"H": 0, "A": 0, "D": 0, "total": 0, "spells_count": 0, "mean_duration": 0.0}
        })

    sorted_frames = sorted(USER_LABELS.keys())
    spells = []
    current_label = None
    current_start = None
    prev_frame = None

    counts = {"H": 0, "A": 0, "D": 0}

    for vf in sorted_frames:
        lbl = USER_LABELS[vf]
        counts[lbl] += 1

        if current_label is None:
            current_label = lbl
            current_start = vf
        elif lbl != current_label or vf != prev_frame + 1:
            # End current spell
            duration_sec = round((prev_frame - current_start + 1) / float(FPS), 2)
            spells.append({
                "start": current_start,
                "end": prev_frame,
                "label": current_label,
                "duration_sec": duration_sec
            })
            current_label = lbl
            current_start = vf
        prev_frame = vf

    if current_label is not None and prev_frame is not None:
        duration_sec = round((prev_frame - current_start + 1) / float(FPS), 2)
        spells.append({
            "start": current_start,
            "end": prev_frame,
            "label": current_label,
            "duration_sec": duration_sec
        })

    total_labeled = sum(counts.values())
    alive_count = counts["H"] + counts["A"]
    possession_spells = [s for s in spells if s["label"] in ("H", "A")]
    mean_dur = round((alive_count / float(FPS) / len(possession_spells)), 2) if possession_spells else 0.0

    return jsonify({
        "spells": spells,
        "stats": {
            "H": counts["H"],
            "A": counts["A"],
            "D": counts["D"],
            "total": total_labeled,
            "spells_count": len(possession_spells),
            "mean_duration": mean_dur
        }
    })


@app.route("/api/ai_fill_range", methods=["POST"])
def api_ai_fill_range():
    """Automatically populate labels using AI heuristic over a range."""
    data = request.get_json(force=True)
    start = int(data.get("start", 11185))
    end = int(data.get("end", 11450))

    if start > end:
        start, end = end, start

    start = max(0, start)
    end = min(TOTAL_FRAMES - 1, end)

    count = 0
    for vf in range(start, end + 1):
        fd = get_single_frame(vf)
        if fd:
            suggested = fd["telemetry"]["ai_suggestion"]["label"]
            USER_LABELS[vf] = suggested
            count += 1

    save_session()
    return jsonify({"success": True, "start": start, "end": end, "filled_count": count})


@app.route("/api/load_reference_gt", methods=["POST"])
def api_load_reference_gt():
    """Copy official 5-minute ground truth into user's session."""
    global USER_LABELS
    for vf, lbl in REFERENCE_GT.items():
        USER_LABELS[vf] = lbl
    save_session()
    return jsonify({"success": True, "loaded_count": len(REFERENCE_GT)})


@app.route("/api/clear_all", methods=["POST"])
def api_clear_all():
    """Clear all user annotations."""
    global USER_LABELS
    USER_LABELS.clear()
    save_session()
    return jsonify({"success": True})


@app.route("/api/export_csv", methods=["POST"])
def api_export_csv():
    """Export all user annotations to competition-compatible CSV."""
    os.makedirs(os.path.dirname(EXPORT_CSV_PATH), exist_ok=True)
    with open(EXPORT_CSV_PATH, "w", encoding="utf-8") as f:
        f.write("frame,label\n")
        for vf in range(TOTAL_FRAMES):
            lbl = USER_LABELS.get(vf, "X")
            f.write(f"{vf},{lbl}\n")

    return jsonify({
        "success": True,
        "export_path": EXPORT_CSV_PATH,
        "labeled_count": len(USER_LABELS)
    })


@app.route("/api/export_training_set", methods=["POST"])
def api_export_training_set():
    """Extract and compile ML-ready training set from labeled frames."""
    if not USER_LABELS:
        return jsonify({"error": "No frames are currently labeled."}), 400

    labels_csv = os.path.join(REPO_ROOT, "data", "custom_training_labels.csv")
    dataset_csv = os.path.join(REPO_ROOT, "data", "custom_training_dataset.csv")
    os.makedirs(os.path.dirname(labels_csv), exist_ok=True)

    sorted_labeled = sorted(USER_LABELS.keys())
    with open(labels_csv, "w", encoding="utf-8") as f:
        f.write("video_frame,possession\n")
        for vf in sorted_labeled:
            f.write(f"{vf},{USER_LABELS[vf]}\n")

    # Merge with pre-extracted features if available
    features_15min = os.path.join(REPO_ROOT, "data", "features_first_15min_play.csv")
    features_public = os.path.join(REPO_ROOT, "data", "features_public_window.csv")
    feat_source = None
    if os.path.exists(features_15min):
        feat_source = features_15min
    elif os.path.exists(features_public):
        feat_source = features_public

    merged_count = 0
    if feat_source:
        try:
            import pandas as pd
            feat_df = pd.read_csv(feat_source)
            lbl_df = pd.DataFrame(list(USER_LABELS.items()), columns=["video_frame", "possession"])
            merged_df = pd.merge(feat_df, lbl_df, on="video_frame", how="inner")
            merged_df.to_csv(dataset_csv, index=False)
            merged_count = len(merged_df)
        except Exception as e:
            print(f"[Labeler] Feature merge error: {e}")

    return jsonify({
        "success": True,
        "labels_file": labels_csv,
        "dataset_file": dataset_csv if merged_count > 0 else None,
        "labeled_count": len(sorted_labeled),
        "merged_features_count": merged_count,
        "min_frame": sorted_labeled[0],
        "max_frame": sorted_labeled[-1]
    })


def initialize():
    """Initialize index and datasets on boot."""
    init_byte_offsets()
    init_reference_gt()
    load_saved_session()


if __name__ == "__main__":
    initialize()
    print("[Labeler] Starting server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
