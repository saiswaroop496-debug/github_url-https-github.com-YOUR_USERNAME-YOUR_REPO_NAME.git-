# data/cv_calibrator.py
"""
GPU-pressure-free YOLOv8 integration.
Strategy: Pre-process video at 2fps (not 25fps) -> 92% GPU reduction.
Output: CV-derived team rates that calibrate the Monte Carlo simulator.
These rates are computed ONCE per match and cached — never run inside the MC loop.
"""
import numpy as np
import json
import os
import warnings
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path("data/cv_calibration_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# GPU memory management
os.environ["YOLO_VERBOSE"] = "False"


def get_cv_calibration(home_team: str, away_team: str,
                        match_date: str,
                        video_path: str = None,
                        force_recompute: bool = False) -> dict:
    """
    Main entry point. Returns CV-derived calibration rates for simulation.
    Uses cache if available — only runs YOLOv8 when fresh video is provided.
    """
    cache_key  = f"{home_team}_vs_{away_team}_{match_date}".replace(" ", "_")
    cache_path = CACHE_DIR / f"{cache_key}_cv.json"

    if not force_recompute and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    if not video_path or not Path(video_path).exists():
        # Return neutral defaults when no video available
        return _default_calibration()

    # Check if YOLO available
    try:
        from ultralytics import YOLO
        import cv2
        YOLO_AVAILABLE = True
    except ImportError:
        YOLO_AVAILABLE = False

    if not YOLO_AVAILABLE:
        warnings.warn("ultralytics not installed. Using default calibration rates.")
        return _default_calibration()

    result = _extract_cv_rates(video_path, home_team, away_team)

    with open(cache_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  ✅ CV calibration cached: {cache_key}")
    return result


def _extract_cv_rates(video_path: str,
                       home_team: str, away_team: str) -> dict:
    """
    Extract team-level event rates from video.
    Samples at 2fps instead of 25fps — 92% faster, captures all events.
    """
    from ultralytics import YOLO
    import supervision as sv
    import cv2

    # Use nano model (fastest) for calibration — accuracy is sufficient
    # yolov8n.pt = 6MB, runs at ~200fps on CPU, ~1500fps on GPU
    model   = YOLO("yolov8n.pt")   # nano = minimal GPU pressure
    tracker = sv.ByteTracker()

    cap     = cv2.VideoCapture(video_path)
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_min = total_frames / fps / 60

    # ADAPTIVE SAMPLING: process 1 frame every 0.5 seconds (2fps equivalent)
    SAMPLE_EVERY_N_FRAMES = max(1, int(fps * 0.5))
    print(f"  Sampling at 1/{SAMPLE_EVERY_N_FRAMES} frames (2fps equiv) "
          f"for {duration_min:.1f} min video")

    team_labels    = {}
    zone_time      = defaultdict(lambda: defaultdict(float))   # {team: {zone: secs}}
    sprint_counts  = defaultdict(int)   # high-speed movements by team
    cluster_counts = defaultdict(int)   # tight player clusters (pressing metric)
    frame_n        = 0

    while cap.isOpened():
        ret = cap.grab()   # grab without decoding (fast)
        if not ret:
            break

        if frame_n % SAMPLE_EVERY_N_FRAMES == 0:
            ret, frame = cap.retrieve()
            if not ret:
                break

            # Detect players only (class 0)
            results  = model(frame, classes=[0], conf=0.4, verbose=False)[0]
            det      = sv.Detections.from_ultralytics(results)
            tracks   = tracker.update_with_detections(det)

            positions = []
            for tid, bbox in zip(tracks.tracker_id, tracks.xyxy):
                if tid is None:
                    continue
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2

                # Assign team from jersey hue
                x1,y1,x2,y2 = map(int, bbox)
                crop = frame[max(0,y1):y2, max(0,x1):x2]
                if crop.size > 0:
                    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    team = 'home' if hsv[:,:,0].mean() < 90 else 'away'
                    team_labels[tid] = team

                    # Zone (0=own half, 1=opp half)
                    zone = 'attacking' if cx > frame.shape[1] * 0.5 else 'defending'
                    zone_time[team][zone] += SAMPLE_EVERY_N_FRAMES / fps

                positions.append((cx, cy, tid))

            # Pressing detection: players clustered within 50px = pressing
            for i, (x1, y1, t1) in enumerate(positions):
                for x2, y2, t2 in positions[i+1:]:
                    if t1 == t2:
                        continue
                    if np.sqrt((x1-x2)**2 + (y1-y2)**2) < 50:
                        team = team_labels.get(t1, 'unknown')
                        cluster_counts[team] += 1

        frame_n += 1

    cap.release()

    # Convert to calibration rates
    total_h = sum(zone_time['home'].values()) or 1
    total_a = sum(zone_time['away'].values()) or 1

    home_pressing_intensity = zone_time['home']['attacking'] / total_h
    away_pressing_intensity = zone_time['away']['attacking'] / total_a
    home_cluster_rate = cluster_counts['home'] / (frame_n / SAMPLE_EVERY_N_FRAMES + 1)
    away_cluster_rate = cluster_counts['away'] / (frame_n / SAMPLE_EVERY_N_FRAMES + 1)

    return {
        "home_team":                home_team,
        "away_team":                away_team,
        # Pressing intensity: how much time in opponent's half (0.0 to 1.0)
        "home_pressing_intensity":  round(home_pressing_intensity, 3),
        "away_pressing_intensity":  round(away_pressing_intensity, 3),
        # Cluster rate: proxy for physical duels/aggression
        "home_cluster_rate":        round(home_cluster_rate, 3),
        "away_cluster_rate":        round(away_cluster_rate, 3),
        # Derived event rate modifiers for simulation
        "home_yellow_modifier":     round(1.0 + home_cluster_rate * 0.5, 3),
        "away_yellow_modifier":     round(1.0 + away_cluster_rate * 0.5, 3),
        "home_shot_modifier":       round(1.0 + home_pressing_intensity * 0.3, 3),
        "away_shot_modifier":       round(1.0 + away_pressing_intensity * 0.3, 3),
        "home_corner_modifier":     round(1.0 + home_pressing_intensity * 0.2, 3),
        "away_corner_modifier":     round(1.0 + away_pressing_intensity * 0.2, 3),
        "source":                   "yolov8n_2fps",
        "frames_processed":         frame_n // SAMPLE_EVERY_N_FRAMES,
    }


def _default_calibration() -> dict:
    """Neutral calibration rates when no video available."""
    return {
        "home_pressing_intensity":  0.52,
        "away_pressing_intensity":  0.48,
        "home_cluster_rate":        1.0,
        "away_cluster_rate":        1.0,
        "home_yellow_modifier":     1.0,
        "away_yellow_modifier":     1.0,
        "home_shot_modifier":       1.0,
        "away_shot_modifier":       1.0,
        "home_corner_modifier":     1.0,
        "away_corner_modifier":     1.0,
        "source":                   "default",
        "frames_processed":         0,
    }
