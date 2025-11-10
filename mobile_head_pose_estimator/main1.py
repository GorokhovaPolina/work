# main.py
import glob
import os
import json
import cv2
from estimator import MobileHeadPoseEstimator
from visualizer import visualize

def main():
    estimator = MobileHeadPoseEstimator(mode='pnp')
    files = sorted(glob.glob("jsons/snapshot_*.json"))
    
    for path in files:
        with open(path) as f:
            data = json.load(f)
        
        result = estimator.process_json(path)
        if not result or 'error' in result:
            continue
        
        img_path = path.replace('.json', '.jpg').replace('jsons', '../mydataset')
        img = cv2.imread(img_path)
        if img is None: continue
        
        nose = tuple(map(int, data['props']['kp_nose_tip']))
        visualize(img, nose, result)
        
        out = path.replace('jsons', 'output').replace('.json', '_vis.jpg')
        cv2.imwrite(out, img)
        print(f"→ {out}")

if __name__ == "__main__":
    main()