#!/usr/bin/env python3
"""
解碼 SAM2 segmentation counts 並產生黑白遮罩
"""

import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
import os
from pathlib import Path

def decode_rle_to_mask(rle_data):
    """
    將 RLE 格式的 segmentation 解碼為二進制遮罩
    
    Args:
        rle_data (dict): 包含 'size' 和 'counts' 的 RLE 數據
        
    Returns:
        numpy.ndarray: 二進制遮罩 (H, W)
    """
    # 構建符合 pycocotools 格式的 RLE 對象
    rle = {
        'size': rle_data['size'],  # [height, width]
        'counts': rle_data['counts']
    }
    
    # 解碼為二進制遮罩
    mask = mask_utils.decode(rle)
    
    return mask

def process_annotations_to_masks(json_file_path, output_dir=None):
    """
    處理 JSON 檔案中的所有 annotations，為每個 shrimp 產生遮罩
    
    Args:
        json_file_path (str): JSON 檔案路徑
        output_dir (str): 輸出目錄，如果為 None 則使用 JSON 檔案所在目錄
    """
    # 載入 JSON 數據
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 取得圖片路徑和尺寸
    image_path = data['image_path']
    img_width = data['img_width']
    img_height = data['img_height']
    
    print(f"處理圖片: {image_path}")
    print(f"圖片尺寸: {img_width} x {img_height}")
    
    # 設定輸出目錄
    if output_dir is None:
        output_dir = os.path.dirname(json_file_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 處理每個 annotation
    for i, annotation in enumerate(data['annotations']):
        class_name = annotation['class_name']
        bbox = annotation['bbox']
        segmentation = annotation['segmentation']
        score = annotation['score'][0] if isinstance(annotation['score'], list) else annotation['score']
        
        print(f"\n處理第 {i+1} 個 {class_name}:")
        print(f"  Bounding Box: {bbox}")
        print(f"  Score: {score:.3f}")
        
        # 解碼 segmentation 為遮罩
        try:
            mask = decode_rle_to_mask(segmentation)
            print(f"  遮罩尺寸: {mask.shape}")
            print(f"  遮罩像素數: {np.sum(mask)} / {mask.size}")
            
            # 將遮罩轉換為 0-255 的圖像 (黑白)
            mask_image = (mask * 255).astype(np.uint8)
            
            # 生成輸出檔名
            base_name = Path(json_file_path).stem
            output_filename = f"{base_name}_{class_name}_{i+1}_mask.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # 儲存遮罩圖像
            cv2.imwrite(output_path, mask_image)
            print(f"  遮罩已儲存: {output_path}")
            
            # 也創建一個彩色版本來顯示遮罩覆蓋
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            colored_mask[mask > 0] = [0, 255, 0]  # 綠色遮罩
            
            colored_output_filename = f"{base_name}_{class_name}_{i+1}_colored_mask.png"
            colored_output_path = os.path.join(output_dir, colored_output_filename)
            cv2.imwrite(colored_output_path, colored_mask)
            print(f"  彩色遮罩已儲存: {colored_output_path}")
            
        except Exception as e:
            print(f"  錯誤: 無法解碼 segmentation - {e}")
    
    # 創建綜合遮罩 (所有物件)
    try:
        combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        for i, annotation in enumerate(data['annotations']):
            segmentation = annotation['segmentation']
            mask = decode_rle_to_mask(segmentation)
            combined_mask = np.maximum(combined_mask, mask * 255)
        
        base_name = Path(json_file_path).stem
        combined_output_filename = f"{base_name}_combined_mask.png"
        combined_output_path = os.path.join(output_dir, combined_output_filename)
        cv2.imwrite(combined_output_path, combined_mask)
        print(f"\n綜合遮罩已儲存: {combined_output_path}")
        
    except Exception as e:
        print(f"\n錯誤: 無法創建綜合遮罩 - {e}")

def main():
    # 處理當前的 JSON 檔案
    json_file = r"d:\Git\Grounded-SAM-2\samples\yolo_sam2_video_demo_enhanced_left\frame_015314\frame_015314_results.json"
    
    if not os.path.exists(json_file):
        print(f"錯誤: 找不到檔案 {json_file}")
        return
    
    print("開始解碼 segmentation masks...")
    process_annotations_to_masks(json_file)
    print("\n完成!")

if __name__ == "__main__":
    main()
