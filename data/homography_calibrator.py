# data/homography_calibrator.py
"""
Interactive homography calibration tool.
Click 4 pitch corner points on a frame → get the H matrix.
Run once per match recording type (same camera = same H).
"""
import cv2
import numpy as np
import json

PITCH_CORNERS = np.float32([
    [0, 0],           # Top-left corner of pitch
    [105, 0],         # Top-right corner
    [105, 68],        # Bottom-right corner
    [0, 68]           # Bottom-left corner
])

clicked_points = []

def click_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([x, y])
        print(f"  Point {len(clicked_points)}: ({x}, {y})")
        if len(clicked_points) == 4:
            print("  4 points selected. Press any key to compute homography.")

def calibrate_from_video(video_path: str,
                          frame_number: int = 0,
                          output_path: str = "data/homography.json"):
    global clicked_points
    clicked_points = []

    cap   = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read frame.")
        return None

    print("Click the 4 corner points of the pitch in this order:")
    print("  1. Top-Left  2. Top-Right  3. Bottom-Right  4. Bottom-Left")

    cv2.namedWindow("Calibrate Homography")
    cv2.setMouseCallback("Calibrate Homography", click_handler)

    while True:
        display = frame.copy()
        for i, pt in enumerate(clicked_points):
            cv2.circle(display, tuple(pt), 5, (0, 255, 0), -1)
            cv2.putText(display, str(i+1), tuple(pt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Calibrate Homography", display)
        if cv2.waitKey(1) & 0xFF == ord('q') or len(clicked_points) == 4:
            if len(clicked_points) == 4:
                break

    cv2.destroyAllWindows()

    src_pts = np.float32(clicked_points)
    H, _    = cv2.findHomography(src_pts, PITCH_CORNERS)

    result = {"H": H.tolist(), "src_points": clicked_points}
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  ✅ Homography saved to {output_path}")
    print(f"  H matrix:\n{H}")
    return H


def load_homography(path: str = "data/homography.json") -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    return np.array(data["H"])
