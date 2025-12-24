import os
import sys
from pathlib import Path

def main():
    json_dir = "aflw2000_json"
    image_dir = "../AFLW2000"
    mat_dir = "../AFLW2000"

    from evaluation.evaluate_aflw2000_3d import (
        load_aflw2000_ground_truth,
        evaluate_pose_estimation,
        print_metrics_table,
        plot_comparison
    )

    gt_data = load_aflw2000_ground_truth(json_dir)

    max_samples = None
    
    metrics, results, errors_detail = evaluate_pose_estimation(
        gt_data, json_dir,
        methods=['pnp', 'geom'],
        max_samples=max_samples,
        save_errors=True
    )

    print_metrics_table(metrics)
    
    output_dir = "evaluation_results"
    plot_comparison(metrics, output_dir)


if __name__ == "__main__":
    main()
