#!/usr/bin/env python3
"""
左右影像物體匹配程式
使用NCC (Normalized Cross-Correlation) 演算法對左右影像的物體進行匹配
"""

import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
import os
from pathlib import Path
import matplotlib.pyplot as plt


def decode_rle_to_mask(rle_data):
    """
    將 RLE 格式的 segmentation 解碼為二進制遮罩
    
    Args:
        rle_data (dict): 包含 'size' 和 'counts' 的 RLE 數據
        
    Returns:
        numpy.ndarray: 二進制遮罩 (H, W)
    """
    rle = {
        'size': rle_data['size'],  # [height, width]
        'counts': rle_data['counts']
    }
    
    # 解碼為二進制遮罩
    mask = mask_utils.decode(rle)
    
    return mask


def load_image_and_annotations(json_path):
    """
    載入JSON檔案和對應的圖像
    
    Args:
        json_path (str): JSON檔案路徑
        
    Returns:
        tuple: (image, annotations_data)
    """
    # 讀取JSON檔案
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 取得圖像路徑 (extracted_frames* 資料夾在根目錄)
    image_path = data['image_path']
    
    # 將反斜線轉換為正斜線，並建構完整路徑 
    # extracted_frames_left 和 extracted_frames_right 在根目錄
    image_path = image_path.replace('\\', os.sep)
    root_dir = r"d:\Git\Grounded-SAM-2"
    full_image_path = os.path.join(root_dir, image_path)
    full_image_path = os.path.normpath(full_image_path)
    
    print(f"嘗試載入圖像: {full_image_path}")
    
    # 載入圖像
    image = cv2.imread(full_image_path)
    if image is None:
        raise FileNotFoundError(f"無法載入圖像: {full_image_path}")
    
    # 轉換為RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image_rgb, data


def extract_object_roi(image, annotation):
    """
    根據annotation從圖像中提取物體的ROI
    
    Args:
        image (numpy.ndarray): 原始圖像
        annotation (dict): 物體標註資訊
        
    Returns:
        tuple: (roi_image, roi_mask, bbox)
    """
    # 解碼遮罩
    mask = decode_rle_to_mask(annotation['segmentation'])
    
    # 取得邊界框
    bbox = annotation['bbox']  # [x1, y1, x2, y2]
    x1, y1, x2, y2 = [int(x) for x in bbox]
    
    # 確保邊界框在圖像範圍內
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # 提取ROI
    roi_image = image[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]
    
    # 應用遮罩到ROI
    masked_roi = roi_image.copy()
    masked_roi[roi_mask == 0] = 0  # 將非物體區域設為黑色
    
    return masked_roi, roi_mask, (x1, y1, x2, y2)


def compute_ncc(template, search_region):
    """
    計算模板和搜索區域之間的正規化交叉相關 (NCC)
    
    Args:
        template (numpy.ndarray): 模板圖像
        search_region (numpy.ndarray): 搜索區域
        
    Returns:
        tuple: (NCC值, 匹配位置) 或 (0.0, (0, 0)) 如果無法匹配
    """
    # 轉換為灰階
    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    if len(search_region.shape) == 3:
        search_region = cv2.cvtColor(search_region, cv2.COLOR_RGB2GRAY)
    
    # 檢查尺寸要求：模板必須小於或等於搜索區域
    template_h, template_w = template.shape
    search_h, search_w = search_region.shape
    
    # 如果模板比搜索區域大，交換它們的角色
    if template_h > search_h or template_w > search_w:
        # 交換模板和搜索區域
        template, search_region = search_region, template
        template_h, template_w = template.shape
        search_h, search_w = search_region.shape
    
    # 再次檢查，如果仍然不符合要求，調整模板大小
    if template_h > search_h or template_w > search_w:
        # 計算縮放比例
        scale_h = search_h / template_h if template_h > search_h else 1.0
        scale_w = search_w / template_w if template_w > search_w else 1.0
        scale = min(scale_h, scale_w, 1.0)
        
        # 調整模板大小
        new_h = max(1, int(template_h * scale))
        new_w = max(1, int(template_w * scale))
        template = cv2.resize(template, (new_w, new_h))
    
    # 如果任一圖像為空或太小，返回低分數
    if template.size == 0 or search_region.size == 0:
        return 0.0, (0, 0)
    
    if template.shape[0] < 3 or template.shape[1] < 3:
        return 0.0, (0, 0)
    
    if search_region.shape[0] < 3 or search_region.shape[1] < 3:
        return 0.0, (0, 0)
    
    try:
        # 使用OpenCV的模板匹配
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        
        # 取得最大值
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        return max_val, max_loc
        
    except Exception as e:
        print(f"    模板匹配異常: {e}")
        return 0.0, (0, 0)


def perform_object_matching(left_data, right_data, left_image, right_image, ncc_threshold=0.3):
    """
    執行左右影像物體匹配
    
    Args:
        left_data (dict): 左影像的標註資料
        right_data (dict): 右影像的標註資料
        left_image (numpy.ndarray): 左影像
        right_image (numpy.ndarray): 右影像
        ncc_threshold (float): NCC匹配閾值
        
    Returns:
        list: 匹配結果列表
    """
    matches = []
    
    # 對每個左影像物體尋找右影像中的匹配
    for i, left_ann in enumerate(left_data['annotations']):
        print(f"\n處理左影像物體 {i+1}/{len(left_data['annotations'])}")
        
        # 提取左影像物體ROI
        left_roi, left_mask, left_bbox = extract_object_roi(left_image, left_ann)
        
        if left_roi.size == 0:
            print(f"  跳過: 左影像物體 {i+1} ROI為空")
            continue
        
        best_match = None
        best_ncc = -1
        
        # 在右影像的所有物體中尋找最佳匹配
        for j, right_ann in enumerate(right_data['annotations']):
            # 提取右影像物體ROI
            right_roi, right_mask, right_bbox = extract_object_roi(right_image, right_ann)
            
            if right_roi.size == 0:
                print(f"    跳過: 右影像物體 {j+1} ROI為空")
                continue
            
            # 顯示ROI尺寸資訊
            print(f"    左ROI尺寸: {left_roi.shape}, 右ROI尺寸: {right_roi.shape}")
            
            # 計算NCC
            try:
                ncc_score, match_loc = compute_ncc(left_roi, right_roi)
                
                print(f"  左物體 {i+1} vs 右物體 {j+1}: NCC = {ncc_score:.3f}")
                
                # 更新最佳匹配
                if ncc_score > best_ncc and ncc_score > ncc_threshold:
                    best_ncc = ncc_score
                    best_match = {
                        'left_idx': i,
                        'right_idx': j,
                        'left_annotation': left_ann,
                        'right_annotation': right_ann,
                        'left_roi': left_roi,
                        'right_roi': right_roi,
                        'left_bbox': left_bbox,
                        'right_bbox': right_bbox,
                        'ncc_score': ncc_score,
                        'match_location': match_loc
                    }
            except Exception as e:
                print(f"  錯誤: 計算NCC失敗 - {e}")
        
        # 儲存匹配結果
        if best_match:
            matches.append(best_match)
            print(f"  ✓ 找到匹配: 左物體 {i+1} <-> 右物體 {best_match['right_idx']+1} (NCC = {best_ncc:.3f})")
        else:
            print(f"  ✗ 未找到匹配: 左物體 {i+1}")
    
    return matches


def visualize_matches(left_image, right_image, matches, output_dir):
    """
    視覺化匹配結果並儲存圖片
    
    Args:
        left_image (numpy.ndarray): 左影像
        right_image (numpy.ndarray): 右影像
        matches (list): 匹配結果
        output_dir (str): 輸出目錄
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 創建匹配結果的整體視覺化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 顯示左右影像
    ax1.imshow(left_image)
    ax1.set_title('Left Image', fontsize=16)
    ax1.axis('off')
    
    ax2.imshow(right_image)
    ax2.set_title('Right Image', fontsize=16)
    ax2.axis('off')
    
    # 在影像上繪製匹配的物體
    colors = plt.cm.tab10(np.linspace(0, 1, len(matches)))
    
    for i, match in enumerate(matches):
        color = colors[i]
        
        # 繪製左影像邊界框
        left_bbox = match['left_bbox']
        x1, y1, x2, y2 = left_bbox
        rect_left = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                 linewidth=3, edgecolor=color, facecolor='none')
        ax1.add_patch(rect_left)
        ax1.text(x1, y1-10, f"Match {i+1}\nNCC: {match['ncc_score']:.3f}", 
                color=color, fontsize=12, fontweight='bold')
        
        # 繪製右影像邊界框
        right_bbox = match['right_bbox']
        x1, y1, x2, y2 = right_bbox
        rect_right = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                  linewidth=3, edgecolor=color, facecolor='none')
        ax2.add_patch(rect_right)
        ax2.text(x1, y1-10, f"Match {i+1}\nNCC: {match['ncc_score']:.3f}", 
                color=color, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # 儲存整體匹配結果
    overall_output_path = os.path.join(output_dir, 'overall_matches.png')
    plt.savefig(overall_output_path, dpi=300, bbox_inches='tight')
    print(f"整體匹配結果已儲存: {overall_output_path}")
    plt.close()
    
    # 為每個匹配創建詳細視覺化
    for i, match in enumerate(matches):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 左影像ROI
        axes[0, 0].imshow(match['left_roi'])
        axes[0, 0].set_title(f'Left Object {match["left_idx"]+1}', fontsize=14)
        axes[0, 0].axis('off')
        
        # 右影像ROI
        axes[0, 1].imshow(match['right_roi'])
        axes[0, 1].set_title(f'Right Object {match["right_idx"]+1}', fontsize=14)
        axes[0, 1].axis('off')
        
        # 左影像完整圖與邊界框
        axes[1, 0].imshow(left_image)
        left_bbox = match['left_bbox']
        x1, y1, x2, y2 = left_bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=3, edgecolor='red', facecolor='none')
        axes[1, 0].add_patch(rect)
        axes[1, 0].set_title('Left Image with Bbox', fontsize=14)
        axes[1, 0].axis('off')
        
        # 右影像完整圖與邊界框
        axes[1, 1].imshow(right_image)
        right_bbox = match['right_bbox']
        x1, y1, x2, y2 = right_bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=3, edgecolor='red', facecolor='none')
        axes[1, 1].add_patch(rect)
        axes[1, 1].set_title('Right Image with Bbox', fontsize=14)
        axes[1, 1].axis('off')
        
        # 添加匹配資訊
        fig.suptitle(f'Match {i+1}: NCC Score = {match["ncc_score"]:.3f}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # 儲存詳細匹配結果
        detail_output_path = os.path.join(output_dir, f'match_{i+1}_detail.png')
        plt.savefig(detail_output_path, dpi=300, bbox_inches='tight')
        print(f"詳細匹配結果 {i+1} 已儲存: {detail_output_path}")
        plt.close()


def save_matches_json(matches, output_path):
    """
    將匹配結果儲存為JSON檔案
    
    Args:
        matches (list): 匹配結果
        output_path (str): 輸出檔案路徑
    """
    # 準備可序列化的匹配資料
    serializable_matches = []
    
    for match in matches:
        serializable_match = {
            'left_object_idx': match['left_idx'],
            'right_object_idx': match['right_idx'],
            'ncc_score': float(match['ncc_score']),
            'left_bbox': match['left_bbox'],
            'right_bbox': match['right_bbox'],
            'left_class_name': match['left_annotation']['class_name'],
            'right_class_name': match['right_annotation']['class_name'],
            'left_score': match['left_annotation']['score'],
            'right_score': match['right_annotation']['score'],
            'match_location': match['match_location']
        }
        serializable_matches.append(serializable_match)
    
    # 儲存JSON檔案
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_matches': len(serializable_matches),
            'matches': serializable_matches
        }, f, indent=2, ensure_ascii=False)
    
    print(f"匹配結果JSON已儲存: {output_path}")


def main():
    # 定義檔案路徑
    left_json_path = r"d:\Git\Grounded-SAM-2\samples\yolo_sam2_video_demo_enhanced_left\frame_015314\frame_015314_results.json"
    right_json_path = r"d:\Git\Grounded-SAM-2\samples\yolo_sam2_video_demo_enhanced_right\frame_015395\frame_015395_results.json"
    
    # 輸出目錄
    output_dir = r"d:\Git\Grounded-SAM-2\object_matching_results"
    
    try:
        print("=== 左右影像物體匹配程式 ===")
        print(f"左影像JSON: {left_json_path}")
        print(f"右影像JSON: {right_json_path}")
        print(f"輸出目錄: {output_dir}")
        
        # 載入左右影像和標註資料
        print("\n1. 載入左影像和標註資料...")
        left_image, left_data = load_image_and_annotations(left_json_path)
        print(f"   左影像尺寸: {left_image.shape}")
        print(f"   左影像物體數量: {len(left_data['annotations'])}")
        
        print("\n2. 載入右影像和標註資料...")
        right_image, right_data = load_image_and_annotations(right_json_path)
        print(f"   右影像尺寸: {right_image.shape}")
        print(f"   右影像物體數量: {len(right_data['annotations'])}")
        
        # 執行物體匹配
        print("\n3. 執行NCC物體匹配...")
        matches = perform_object_matching(left_data, right_data, left_image, right_image)
        
        print(f"\n4. 匹配完成! 共找到 {len(matches)} 個匹配")
        
        if len(matches) > 0:
            # 視覺化匹配結果
            print("\n5. 產生視覺化結果...")
            visualize_matches(left_image, right_image, matches, output_dir)
            
            # 儲存匹配結果JSON
            print("\n6. 儲存匹配結果...")
            json_output_path = os.path.join(output_dir, 'matching_results.json')
            save_matches_json(matches, json_output_path)
            
            print("\n=== 匹配結果摘要 ===")
            for i, match in enumerate(matches):
                print(f"匹配 {i+1}: 左物體{match['left_idx']+1} <-> 右物體{match['right_idx']+1} (NCC: {match['ncc_score']:.3f})")
        else:
            print("\n未找到任何匹配的物體")
        
        print(f"\n程式執行完成! 結果已儲存至: {output_dir}")
        
    except Exception as e:
        print(f"錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
