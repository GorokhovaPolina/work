import os
import json

def create_json_for_images_simple():
    image_dir = "C:/Users/polina/source/repos/work/mobile_head_pose_estimator/imgs_Nastya"
    json_dir = "C:/Users/polina/source/repos/work/mobile_head_pose_estimator/markup_Nastya"
    os.makedirs(json_dir, exist_ok=True)
    img_exts = ('.jpg')
    for file in os.listdir(image_dir):
        if file.lower().endswith(img_exts):
            json_name = os.path.splitext(file)[0] + '.json'
            json_path = os.path.join(json_dir, json_name)
            data = {
                "head_pose": {
                    "yaw": 0.0,
                    "pitch": 0.0,
                    "roll": 0.0
                    }
            }
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Создан: {json_name}")

create_json_for_images_simple()
