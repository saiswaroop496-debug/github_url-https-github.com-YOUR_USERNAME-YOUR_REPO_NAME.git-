import numpy as np
from collections import defaultdict

# Divide pitch into 18 zones (6 columns × 3 rows)
ZONES_X = 6
ZONES_Y = 3
PITCH_L = 105.0
PITCH_W = 68.0

def get_zone(x_m: float, y_m: float) -> tuple:
    """Return (col, row) zone index for a pitch position in meters."""
    try:
        col = max(0, min(ZONES_X - 1, int(max(0, float(x_m)) / PITCH_L * ZONES_X)))
        row = max(0, min(ZONES_Y - 1, int(max(0, float(y_m)) / PITCH_W * ZONES_Y)))
        return col, row
    except (ValueError, TypeError):
        return 0, 0


def zone_to_label(col: int, row: int) -> str:
    col = max(0, min(ZONES_X - 1, col))
    row = max(0, min(ZONES_Y - 1, row))
    col_names = ['Own Box', 'Own Third', 'Own Mid', 'Opp Mid', 'Opp Third', 'Opp Box']
    row_names = ['Left', 'Centre', 'Right']
    return f"{col_names[col]} {row_names[row]}"


class ZoneAwareTracker:
    """
    Extends base PlayerMovementTracker with:
    - Zone occupancy heatmaps (where are players spending time?)
    - Pressure index (how many defenders within 5m of ball carrier?)
    - Formation inference (most common defensive shape 4-4-2, 4-3-3, etc.)
    - PPDA proxy (Passes Allowed Per Defensive Action)
    """
    def __init__(self):
        self.zone_time  = defaultdict(lambda: defaultdict(float))  # {team: {zone: seconds}}
        self.positions  = defaultdict(list)   # {track_id: [(x,y,t,team)]}
        self.ball_pos   = []                  # [(x,y,t)] if ball detected

    def update_frame(self, tracks, team_labels: dict,
                      H: np.ndarray, fps: float):
        """Process a single frame — update zones and positions."""
        import cv2
        for tid, bbox in zip(tracks.tracker_id, tracks.xyxy):
            if tid is None:
                continue
            foot_px = np.array([(bbox[0]+bbox[2])/2, bbox[3]])
            pt = np.array([[[float(foot_px[0]), float(foot_px[1])]]], dtype=np.float32)
            pos = cv2.perspectiveTransform(pt, H)[0][0]
            team = team_labels.get(tid, 'unknown')
            zone = get_zone(pos[0], pos[1])
            self.zone_time[team][zone] += 1.0 / fps
            self.positions[tid].append((pos[0], pos[1], team))

    def get_zone_features(self) -> dict:
        """
        Aggregate zone features for model input.
        Returns features like:
        - home_final_third_time: seconds spent in attacking third
        - home_own_box_time: defensive pressure time
        - zone_dominance: which team controls central midfield
        """
        feats = {}
        for team in ['home', 'away']:
            zones = self.zone_time[team]
            # Attacking third = cols 4,5
            att_third = sum(v for (c,r),v in zones.items() if c >= 4)
            # Own box = col 0
            own_box = sum(v for (c,r),v in zones.items() if c == 0)
            # Central midfield = cols 2,3, rows 1 (centre)
            central = sum(v for (c,r),v in zones.items() if c in [2,3] and r == 1)
            total = sum(zones.values()) or 1

            feats[f'{team}_att_third_pct']  = att_third / total
            feats[f'{team}_own_box_pct']    = own_box / total
            feats[f'{team}_central_ctrl']   = central / total

        feats['zone_dominance'] = (
            feats.get('home_central_ctrl', 0) - feats.get('away_central_ctrl', 0)
        )
        return feats

    def estimate_formation(self, team: str, last_n_frames: int = 300) -> str:
        """
        Infer formation from player x-positions.
        Clusters defenders/midfielders/attackers by average x position.
        """
        team_pos = [
            p for tid, positions in self.positions.items()
            for p in positions[-last_n_frames:]
            if len(positions) > 10 and p[2] == team
        ]
        if len(team_pos) < 20:
            return "Unknown"

        xs = [p[0] for p in team_pos]
        # Bin into thirds
        d_count = sum(1 for x in xs if x < PITCH_L * 0.33)
        m_count = sum(1 for x in xs if PITCH_L * 0.33 <= x < PITCH_L * 0.66)
        a_count = sum(1 for x in xs if x >= PITCH_L * 0.66)
        total   = d_count + m_count + a_count or 1

        # Normalize to 10 outfield players
        d = round(d_count / total * 10)
        m = round(m_count / total * 10)
        a = round(a_count / total * 10)
        return f"{d}-{m}-{a}"
