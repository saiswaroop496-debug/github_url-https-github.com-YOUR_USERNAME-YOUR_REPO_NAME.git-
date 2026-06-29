import json, warnings, numpy as np
from pathlib import Path
from collections import defaultdict

TRACKING_DEPS_AVAILABLE = False
_deps_checked = False

def _check_tracking_deps():
    global TRACKING_DEPS_AVAILABLE, _deps_checked
    if _deps_checked:
        return TRACKING_DEPS_AVAILABLE
    _deps_checked = True
    try:
        import cv2
        from ultralytics import YOLO
        import supervision as sv
        TRACKING_DEPS_AVAILABLE = True
    except ImportError:
        warnings.warn(
            "Player tracking deps not installed. "
            "Install locally with: pip install ultralytics supervision opencv-python\n"
            "Streamlit Cloud deployment will use pre-computed movement_stats.json only."
        )
    return TRACKING_DEPS_AVAILABLE


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M  = 68.0
MOVEMENT_STATS_PATH = Path("data/movement_stats.json")


def get_pitch_homography(src_points: np.ndarray) -> np.ndarray:
    import cv2
    dst = np.float32([[0,0],[PITCH_LENGTH_M,0],
                       [PITCH_LENGTH_M,PITCH_WIDTH_M],[0,PITCH_WIDTH_M]])
    H, _ = cv2.findHomography(src_points.astype(np.float32), dst)
    return H


def pixel_to_meters(pt_px: np.ndarray, H: np.ndarray) -> np.ndarray:
    import cv2
    pt = np.array([[[float(pt_px[0]), float(pt_px[1])]]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, H)[0][0]


class PlayerMovementTracker:
    """
    YOLOv8 + ByteTrack player tracker adapted from kazmifactor/Car_Speed_Estimation.
    Requires ultralytics and supervision (local only).
    """
    def __init__(self, model_path="yolov8m.pt", homography_points=None, fps=25.0):
        if not _check_tracking_deps():
            raise RuntimeError(
                "Install tracking deps first: pip install ultralytics supervision opencv-python"
            )
        import supervision as sv
        from ultralytics import YOLO
        self.model    = YOLO(model_path)
        self.tracker  = sv.ByteTracker()
        self.fps      = fps
        self.H        = get_pitch_homography(homography_points) if homography_points is not None else None
        self.positions   = defaultdict(list)
        self.speeds      = defaultdict(list)
        self.team_labels = {}

    def _assign_team_by_jersey(self, crop: np.ndarray) -> str:
        import cv2
        if crop.size == 0:
            return 'unknown'
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_h = hsv[:,:,0].mean()
        return 'home' if mean_h < 90 else 'away'

    def process_video(self, video_path: str, max_frames: int = None) -> dict:
        import cv2
        import supervision as sv
        cap = cv2.VideoCapture(video_path)
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        frame_n = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_n > max_frames):
                break

            results    = self.model(frame, classes=[0], conf=0.4, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            tracks     = self.tracker.update_with_detections(detections)

            for tid, bbox in zip(tracks.tracker_id, tracks.xyxy):
                if tid is None:
                    continue
                foot_px = np.array([(bbox[0]+bbox[2])/2, bbox[3]])
                pos_m   = pixel_to_meters(foot_px, self.H) if self.H is not None else foot_px
                self.positions[tid].append((pos_m[0], pos_m[1], frame_n))

                if len(self.positions[tid]) >= 2:
                    prev, curr = self.positions[tid][-2], self.positions[tid][-1]
                    dt = (curr[2]-prev[2]) / actual_fps
                    if dt > 0:
                        dist  = np.sqrt((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2)
                        self.speeds[tid].append(dist / dt)

                if tid not in self.team_labels:
                    x1,y1,x2,y2 = map(int, bbox)
                    crop = frame[max(0,y1):y2, max(0,x1):x2]
                    self.team_labels[tid] = self._assign_team_by_jersey(crop)

            frame_n += 1
        cap.release()
        return self._compile_stats(actual_fps)

    def _compile_stats(self, fps: float) -> dict:
        team_data = {'home': [], 'away': []}
        for tid, speeds_list in self.speeds.items():
            if len(speeds_list) < 10:
                continue
            s = np.array(speeds_list)
            pos = np.array([(p[0],p[1]) for p in self.positions[tid]])
            dist = float(np.sqrt(np.diff(pos, axis=0)**2).sum())
            player = {
                "mean_speed_ms": float(s.mean()),
                "max_speed_ms":  float(s.max()),
                "distance_m":    dist,
                "sprint_count":  int((s > 7.0).sum()),
                "hi_runs":       int((s > 5.5).sum()),
            }
            team = self.team_labels.get(tid, 'unknown')
            if team in team_data:
                team_data[team].append(player)

        result = {}
        for team in ['home','away']:
            players = team_data[team]
            if not players:
                continue
            result[f"{team}_avg_speed"]       = round(np.mean([p['mean_speed_ms'] for p in players]), 3)
            result[f"{team}_total_distance_m"]= round(sum(p['distance_m'] for p in players), 1)
            result[f"{team}_total_sprints"]   = sum(p['sprint_count'] for p in players)
        result['speed_diff'] = result.get('home_avg_speed',0) - result.get('away_avg_speed',0)
        return result


def process_match_videos(video_dir: str, homography_points=None):
    """
    Batch process all .mp4 files in video_dir.
    File naming: "Brazil_vs_Germany_2026-06-14.mp4"
    Output: data/movement_stats.json
    """
    if not _check_tracking_deps():
        print("Tracking deps not available. Skipping video processing.")
        return {}

    results = {}
    for video_file in Path(video_dir).glob("*.mp4"):
        parts = video_file.stem.split("_vs_")
        if len(parts) != 2:
            continue
        home = parts[0]
        rest = parts[1].split("_")
        away, date = rest[0], rest[1] if len(rest) > 1 else "unknown"
        key = f"{home}_vs_{away}_{date}"
        print(f"📹 Processing: {video_file.name}")
        try:
            tracker = PlayerMovementTracker(homography_points=homography_points)
            stats   = tracker.process_video(str(video_file))
            results[key] = stats
            print(f"  ✅ {key}: {stats}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

    MOVEMENT_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MOVEMENT_STATS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Saved → {MOVEMENT_STATS_PATH}")
    return results


def fallback_empty_movement_stats():
    """Return zero-filled stats when no video or tracking deps are available."""
    return {
        "home_avg_speed": 0.0, "away_avg_speed": 0.0,
        "home_total_distance_m": 0.0, "away_total_distance_m": 0.0,
        "home_total_sprints": 0, "away_total_sprints": 0,
        "speed_diff": 0.0
    }


if __name__ == "__main__":
    # Test graceful fallback with no video
    stats = fallback_empty_movement_stats()
    print("Fallback stats:", stats)
    print("Graceful fallback: ✅")
