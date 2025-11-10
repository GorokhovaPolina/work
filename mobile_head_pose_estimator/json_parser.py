import json
from typing import Optional, Dict, Tuple

def load_keypoints_from_json(json_path: str) -> Optional[Dict]:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] {json_path}: {e}")
        return None

    props = data.get("props", {})
    img_size = data.get("image_size", [1, 1])
    if not props:
        return None

    try:
        # Среднее между inner/outer
        def avg(k1, k2):
            p1 = props[k1]
            p2 = props[k2]
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        return {
            'landmarks': {
                'nose': tuple(props["kp_nose_tip"]),
                'left_eye': avg("kp_eye_left_inner", "kp_eye_left_outer"),
                'right_eye': avg("kp_eye_right_inner", "kp_eye_right_outer"),
                'mouth': avg("kp_mouth_left", "kp_mouth_right")
            },
            'image_size': tuple(img_size)
        }
    except KeyError as e:
        print(f"[ERROR] Missing key: {e}")
        return None