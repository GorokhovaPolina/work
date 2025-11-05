import json
from typing import Optional, Dict, Tuple

def load_keypoints_from_json(json_path: str) -> Optional[Dict[str, Tuple[int, int]]]:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read {json_path}: {e}")
        return None

    props = data.get("props", {})
    if not props:
        print(f"[WARN] No 'props' in {json_path}")
        return None

    try:
        nose = tuple(props["kp_nose_tip"])

        left_eye = (
            (props["kp_eye_left_inner"][0] + props["kp_eye_left_outer"][0]) // 2,
            (props["kp_eye_left_inner"][1] + props["kp_eye_left_outer"][1]) // 2
        )
        right_eye = (
            (props["kp_eye_right_inner"][0] + props["kp_eye_right_outer"][0]) // 2,
            (props["kp_eye_right_inner"][1] + props["kp_eye_right_outer"][1]) // 2
        )
        mouth = (
            (props["kp_mouth_left"][0] + props["kp_mouth_right"][0]) // 2,
            (props["kp_mouth_left"][1] + props["kp_mouth_right"][1]) // 2
        )

        return {
            'nose': nose,
            'left_eye': left_eye,
            'right_eye': right_eye,
            'mouth': mouth
        }
    except KeyError as e:
        print(f"[ERROR] Missing key {e} in {json_path}")
        return None
