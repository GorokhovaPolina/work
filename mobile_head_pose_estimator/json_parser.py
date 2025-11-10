import json

def load_keypoints_from_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        w, h = data["image_size"]
        props = data["props"]

        # Средние точки глаз
        left_eye = (
            (props["kp_eye_left_inner"][0] + props["kp_eye_left_outer"][0]) / 2,
            (props["kp_eye_left_inner"][1] + props["kp_eye_left_outer"][1]) / 2
        )
        right_eye = (
            (props["kp_eye_right_inner"][0] + props["kp_eye_right_outer"][0]) / 2,
            (props["kp_eye_right_inner"][1] + props["kp_eye_right_outer"][1]) / 2
        )
        mouth = (
            (props["kp_mouth_left"][0] + props["kp_mouth_right"][0]) / 2,
            (props["kp_mouth_left"][1] + props["kp_mouth_right"][1]) / 2
        )

        return {
            'image_size': (w, h),
            'landmarks': {
                'nose': tuple(props["kp_nose_tip"]),
                'left_eye': left_eye,
                'right_eye': right_eye,
                'mouth': mouth
            }
        }
    except Exception as e:
        print(f"[ERROR] Парсинг JSON: {json_path} → {e}")
        return None