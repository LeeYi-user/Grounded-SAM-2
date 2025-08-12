import json
import math
import os
import sys
import argparse
from typing import Dict, Any, List
import cv2
import numpy as np
from pathlib import Path


def depth_from_disparity(disparity: float, baseline_cm: float = 25.0) -> float:
    """
    Compute depth (cm) from disparity using the provided formula:
    depth = (baseline_cm / 7.6) * (0.96938468^(disparity - 172.87387593) + 19.6022959)
    """
    if disparity is None or disparity <= 0:
        raise ValueError("Disparity must be positive.")
    return (baseline_cm / 7.6) * (0.96938468**(disparity - 172.87387593) + 19.6022959)


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


def load_image_and_annotations(disparity_results_path: str):
    """
    載入視差分析結果和對應的圖像
    
    Args:
        disparity_results_path (str): 視差分析結果JSON檔案路徑
        
    Returns:
        tuple: (image_rgb, image_bgr, disparity_data)
    """
    with open(disparity_results_path, 'r', encoding='utf-8') as f:
        disparity_data = json.load(f)
    
    # 嘗試從JSON檔案中獲取左影像路徑
    left_image_path = disparity_data.get('left_image_path')
    
    if left_image_path and os.path.exists(left_image_path):
        print(f"使用JSON中記錄的左影像路徑: {left_image_path}")
    else:
        print("JSON中未找到左影像路徑，嘗試自動尋找...")
        # 尋找對應的原始圖像檔案
        root_dir = Path(__file__).parent
        
        # 首先嘗試找第一張圖像
        left_frames_dir = root_dir / "extracted_frames_left"
        if left_frames_dir.exists():
            # 找到第一張圖像檔案
            image_files = sorted(left_frames_dir.glob("*.jpg"))
            if image_files:
                left_image_path = str(image_files[0])  # 使用第一張圖像
        
        # 如果沒找到，嘗試其他可能的路徑
        if left_image_path is None or not os.path.exists(left_image_path):
            possible_paths = [
                "demo_images/left_image.jpg",
                "samples/left_sample.jpg",
                "extracted_frames_left/frame_021805.jpg"  # 第一張可用的圖像
            ]
            
            for path in possible_paths:
                full_path = root_dir / path
                if full_path.exists():
                    left_image_path = str(full_path)
                    break
    
    if left_image_path is None or not os.path.exists(left_image_path):
        raise FileNotFoundError("找不到對應的圖像檔案")
    
    print(f"載入圖像: {left_image_path}")
    
    # 載入圖像
    image_bgr = cv2.imread(left_image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"無法載入圖像: {left_image_path}")
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    return image_rgb, image_bgr, disparity_data


def draw_shrimp_length_on_image(image_bgr, disparity_data, length_results, output_path: str):
    """
    在圖像上繪製蝦子體長資訊
    
    Args:
        image_bgr: OpenCV格式的圖像 (BGR)
        disparity_data: 視差分析資料
        length_results: 體長計算結果
        output_path: 輸出圖像路徑
    """
    # 複製圖像避免修改原始圖像
    annotated_image = image_bgr.copy()
    
    # 建立shrimp_id到體長的對應關係
    length_dict = {}
    for result in length_results:
        shrimp_id = result.get("shrimp_id")
        total_length = result.get("total_length_cm", 0.0)
        length_dict[shrimp_id] = total_length
    
    # 為每隻蝦子繪製資訊
    for analysis_result in disparity_data.get("analysis_results", []):
        shrimp_id = analysis_result.get("shrimp_id")
        left_bbox = analysis_result.get("left_bbox", [])
        
        if len(left_bbox) == 4 and shrimp_id in length_dict:
            x1, y1, x2, y2 = left_bbox
            total_length = length_dict[shrimp_id]
            
            # 繪製邊界框
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 準備文字標籤
            label = f"Shrimp {shrimp_id}: {total_length:.2f} cm"
            
            # 設定文字參數
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            text_color = (255, 255, 255)  # 白色文字
            bg_color = (0, 0, 0)  # 黑色背景
            
            # 獲取文字大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # 計算文字位置（在邊界框上方）
            text_x = x1
            text_y = y1 - 10
            
            # 確保文字不會超出圖像邊界
            if text_y - text_height < 0:
                text_y = y2 + text_height + 10
            
            # 繪製文字背景矩形
            cv2.rectangle(
                annotated_image,
                (text_x, text_y - text_height - 5),
                (text_x + text_width + 10, text_y + 5),
                bg_color,
                -1
            )
            
            # 繪製文字
            cv2.putText(
                annotated_image,
                label,
                (text_x + 5, text_y),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA
            )
            
            # 繪製採樣點
            left_points = analysis_result.get("left_sampled_points", {})
            x_points = left_points.get("x", [])
            y_points = left_points.get("y", [])
            
            for i, (x, y) in enumerate(zip(x_points, y_points)):
                # 繪製採樣點
                cv2.circle(annotated_image, (int(x), int(y)), 3, (0, 0, 255), -1)
                
                # 繪製點的編號
                point_label = str(i + 1)
                cv2.putText(
                    annotated_image,
                    point_label,
                    (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA
                )
            
            # 連接採樣點形成曲線
            if len(x_points) > 1:
                points = [(int(x), int(y)) for x, y in zip(x_points, y_points)]
                for i in range(len(points) - 1):
                    cv2.line(annotated_image, points[i], points[i + 1], (255, 0, 0), 2)
    
    # 在圖像上添加總體資訊
    info_text = f"Total Shrimps: {len(length_results)}"
    cv2.putText(
        annotated_image,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )
    
    # 保存標註後的圖像
    cv2.imwrite(output_path, annotated_image)
    print(f"標註圖像已保存至: {output_path}")


def main():
    # Defaults
    default_input = os.path.join("shrimp_disparity_analysis_results", "shrimp_disparity_results.json")
    default_output = os.path.join("shrimp_disparity_analysis_results", "shrimp_body_lengths.json")

    parser = argparse.ArgumentParser(description="Compute shrimp body segment lengths from disparity JSON.")
    parser.add_argument("-i", "--input", default=default_input, help="Path to input JSON file.")
    parser.add_argument("-o", "--output", default=default_output, help="Path to output JSON file.")
    parser.add_argument("-b", "--baseline", type=float, default=25.0, help="Stereo baseline in cm (default: 25.0)")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization with shrimp lengths on image")
    parser.add_argument("--vis-output", default=None, help="Path for visualization output image")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    baseline_cm = float(args.baseline)
    visualize = args.visualize
    vis_output = args.vis_output

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
    
    # 生成可視化圖像
    if visualize:
        try:
            # 如果未指定可視化輸出路徑，使用預設路徑
            if vis_output is None:
                vis_output = os.path.join(
                    os.path.dirname(output_path), 
                    "shrimp_lengths_visualization.jpg"
                )
            
            # 載入圖像和視差資料
            image_rgb, image_bgr, disparity_data = load_image_and_annotations(input_path)
            
            # 繪製體長資訊到圖像上
            draw_shrimp_length_on_image(image_bgr, disparity_data, results_out, vis_output)
            
        except Exception as e:
            print(f"可視化過程發生錯誤: {str(e)}")
            print("請確認圖像檔案存在且路徑正確")


if __name__ == "__main__":
    main()
