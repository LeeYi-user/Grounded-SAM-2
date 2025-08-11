import json
import math
import os
import sys
import argparse
from typing import Dict, Any, List


def depth_from_disparity(disparity: float, baseline_cm: float = 25.0) -> float:
    """
    Compute depth (cm) from disparity using the provided formula:
    depth = 1358.80552 * (baseline / 7.6) / disparity^0.4686624
    """
    if disparity is None or disparity <= 0:
        raise ValueError("Disparity must be positive.")
    K = 1358.80552 * (baseline_cm / 7.6)
    return K / (disparity ** 0.4686624)


def compute_lengths_for_shrimp(entry: Dict[str, Any], baseline_cm: float = 25.0) -> Dict[str, Any]:
    """
    Given one shrimp analysis entry from the JSON, compute per-segment lengths and total length.

    d = sqrt(c^2 + length^2)
    where
    - c = |a - b|, a = depth(disp2), b = depth(disp1)
    - length = r * baseline / max(disp1, disp2)
      r = sqrt(|x[i+1] - x[i]|^2 + |y[i+1] - y[i]|^2) in pixels (left image Euclidean distance)
    """
    disparities: List[float] = entry.get("disparities", [])
    left_pts = entry.get("left_sampled_points", {})
    left_x: List[float] = left_pts.get("x", [])
    left_y: List[float] = left_pts.get("y", [])

    n = min(len(disparities), len(left_x), len(left_y))
    if n < 2:
        return {
            "shrimp_id": entry.get("shrimp_id"),
            "segment_lengths_cm": [],
            "total_length_cm": 0.0,
            "notes": "Insufficient points to compute segments"
        }

    segment_lengths: List[float] = []
    for i in range(n - 1):
        disp1 = disparities[i]
        disp2 = disparities[i + 1]

        # Depths (cm)
        a = depth_from_disparity(disp2, baseline_cm=baseline_cm)
        b = depth_from_disparity(disp1, baseline_cm=baseline_cm)
        c = abs(a - b)

        # Euclidean distance along x and y (cm)
        r_px = math.hypot(abs(left_x[i + 1] - left_x[i]), abs(left_y[i + 1] - left_y[i]))
        max_disp = max(disp1, disp2)
        if max_disp <= 0:
            raise ValueError("Encountered non-positive disparity when computing length.")
        length_cm = (r_px * baseline_cm) / max_disp

        # body segment length (cm)
        d_cm = math.hypot(c, length_cm)
        segment_lengths.append(d_cm)

    total_length = float(sum(segment_lengths))
    return {
        "shrimp_id": entry.get("shrimp_id"),
        "segment_lengths_cm": segment_lengths,
        "total_length_cm": total_length,
    }


def main():
    # Defaults
    default_input = os.path.join("shrimp_disparity_analysis_results", "shrimp_disparity_results.json")
    default_output = os.path.join("shrimp_disparity_analysis_results", "shrimp_body_lengths.json")

    parser = argparse.ArgumentParser(description="Compute shrimp body segment lengths from disparity JSON.")
    parser.add_argument("-i", "--input", default=default_input, help="Path to input JSON file.")
    parser.add_argument("-o", "--output", default=default_output, help="Path to output JSON file.")
    parser.add_argument("-b", "--baseline", type=float, default=25.0, help="Stereo baseline in cm (default: 25.0)")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    baseline_cm = float(args.baseline)

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results_out: List[Dict[str, Any]] = []
    for entry in data.get("analysis_results", []):
        res = compute_lengths_for_shrimp(entry, baseline_cm=baseline_cm)
        results_out.append(res)

    summary = {
    "source": os.path.relpath(input_path).replace("\\", "/"),
        "baseline_cm": baseline_cm,
        "results": results_out,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Console summary
    print("Computed shrimp body lengths (cm):")
    for res in results_out:
        sid = res.get("shrimp_id")
        total = res.get("total_length_cm")
        segs = res.get("segment_lengths_cm", [])
        print(f"- shrimp_id={sid}: total={total:.3f} cm; segments={[round(x,3) for x in segs]}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
