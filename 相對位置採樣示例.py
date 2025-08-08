#!/usr/bin/env python3
"""
相對位置採樣示例程式 - 使用真實蝦子數據
用真實的蝦子分割資料說明如何在蝦子曲線上進行相對位置採樣
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from pycocotools import mask as mask_utils
import os
from pathlib import Path
from scipy.optimize import minimize

# 解決中文顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_real_shrimp_data(json_path):
    """
    載入真實蝦子資料
    
    Args:
        json_path (str): JSON檔案路徑
        
    Returns:
        tuple: (image, shrimp_masks, shrimp_data)
    """
    # 讀取JSON檔案
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 取得圖像路徑
    image_path = data['image_path']
    image_path = image_path.replace('\\', os.sep)
    root_dir = r"d:\Git\Grounded-SAM-2"
    full_image_path = os.path.join(root_dir, image_path)
    full_image_path = os.path.normpath(full_image_path)
    
    print(f"載入圖像: {full_image_path}")
    
    # 載入圖像
    image = cv2.imread(full_image_path)
    if image is None:
        print(f"警告: 無法載入圖像: {full_image_path}")
        return None, [], []
    
    # 轉換為RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 過濾出蝦子標註
    shrimp_annotations = [ann for ann in data['annotations'] if 'shrimp' in ann['class_name'].lower()]
    
    shrimp_masks = []
    shrimp_data = []
    
    for ann in shrimp_annotations:
        # 解碼遮罩
        if 'segmentation' in ann and ann['segmentation']:
            rle_data = ann['segmentation']
            mask = mask_utils.decode({
                'size': rle_data['size'],
                'counts': rle_data['counts']
            })
            shrimp_masks.append(mask)
            shrimp_data.append(ann)
    
    return image_rgb, shrimp_masks, shrimp_data

def get_mask_points(mask):
    """
    從二進制遮罩中提取所有為 True 的像素點座標
    
    Args:
        mask (numpy.ndarray): 二進制遮罩
        
    Returns:
        tuple: (x_coords, y_coords) 像素點座標陣列
    """
    y_coords, x_coords = np.where(mask > 0)
    return x_coords, y_coords

def extract_object_roi(image, mask):
    """
    從遮罩中提取物體的ROI (Region of Interest)
    
    Args:
        image (numpy.ndarray): 輸入影像
        mask (numpy.ndarray): 二進制遮罩
        
    Returns:
        tuple: (roi, bbox) ROI影像和邊界框
    """
    # 找到遮罩的邊界框
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None, None
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # 提取ROI
    roi = image[y_min:y_max+1, x_min:x_max+1]
    bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    
    return roi, bbox

def compute_ncc(template, image):
    """
    計算標準化交叉相關 (Normalized Cross Correlation)
    
    Args:
        template (numpy.ndarray): 模板影像
        image (numpy.ndarray): 搜尋影像
        
    Returns:
        tuple: (max_ncc_score, best_match_location)
    """
    if template.size == 0 or image.size == 0:
        return 0, (0, 0)
    
    # 轉換為灰階
    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    else:
        template_gray = template
        
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image
    
    # 執行模板匹配
    result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    # 找到最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    return max_val, max_loc

def find_matching_shrimps(left_masks, right_masks, left_image, right_image, ncc_threshold=0.3):
    """
    使用NCC匹配左右影像中的蝦子
    
    Args:
        left_masks (list): 左影像蝦子遮罩列表
        right_masks (list): 右影像蝦子遮罩列表
        left_image (numpy.ndarray): 左影像
        right_image (numpy.ndarray): 右影像
        ncc_threshold (float): NCC匹配閾值
        
    Returns:
        list: 匹配結果 [(left_mask, right_mask, ncc_score), ...]
    """
    matches = []
    
    print(f"尋找蝦子匹配: 左{len(left_masks)}隻, 右{len(right_masks)}隻")
    
    for i, left_mask in enumerate(left_masks):
        # 提取左蝦子ROI
        left_roi, left_bbox = extract_object_roi(left_image, left_mask)
        if left_roi is None:
            continue
        
        best_match = None
        best_ncc = -1
        
        # 在右影像中尋找最佳匹配
        for j, right_mask in enumerate(right_masks):
            # 提取右蝦子ROI
            right_roi, right_bbox = extract_object_roi(right_image, right_mask)
            if right_roi is None:
                continue
            
            # 計算NCC
            try:
                ncc_score, match_loc = compute_ncc(left_roi, right_roi)
                print(f"  左蝦子{i+1} vs 右蝦子{j+1}: NCC = {ncc_score:.3f}")
                
                if ncc_score > best_ncc and ncc_score > ncc_threshold:
                    best_ncc = ncc_score
                    best_match = (left_mask, right_mask, ncc_score)
                    
            except Exception as e:
                print(f"  匹配計算錯誤: {e}")
        
        if best_match:
            matches.append(best_match)
            print(f"  ✓ 找到匹配: 左蝦子{i+1} (NCC = {best_ncc:.3f})")
        else:
            print(f"  ✗ 未找到匹配: 左蝦子{i+1}")
    
    return matches

def fit_quadratic_curve(x_points, y_points):
    """
    擬合二次曲線到點集
    
    Args:
        x_points (array): 點的 x 座標
        y_points (array): 點的 y 座標
        
    Returns:
        array: 二次曲線參數 [a, b, c] for y = ax² + bx + c
    """
    if len(x_points) < 3:
        return np.array([0, 0, np.mean(y_points) if len(y_points) > 0 else 0])
    
    def objective(params):
        a, b, c = params
        y_pred = a * x_points**2 + b * x_points + c
        return np.sum((y_points - y_pred)**2)
    
    # 初始猜測
    initial_guess = [0, 0, np.mean(y_points)]
    
    try:
        result = minimize(objective, initial_guess, method='BFGS')
        return result.x
    except:
        return np.array([0, 0, np.mean(y_points)])

def demonstrate_real_data_sampling():
    """演示使用真實蝦子資料的相對位置採樣"""
    
    print("=== 真實蝦子資料相對位置採樣示例 ===\n")
    
    # 定義左右影像的JSON檔案路徑
    left_json_path = r"d:\Git\Grounded-SAM-2\samples\yolo_sam2_video_demo_enhanced_left\frame_015314\frame_015314_results.json"
    right_json_path = r"d:\Git\Grounded-SAM-2\samples\yolo_sam2_video_demo_enhanced_right\frame_015395\frame_015395_results.json"
    
    # 檢查檔案是否存在
    if not os.path.exists(left_json_path):
        print(f"左影像JSON檔案不存在: {left_json_path}")
        print("使用模擬資料進行示例...")
        demonstrate_simulated_sampling()
        return
    
    if not os.path.exists(right_json_path):
        print(f"右影像JSON檔案不存在: {right_json_path}")
        print("使用模擬資料進行示例...")
        demonstrate_simulated_sampling()
        return
    
    # 載入左右影像資料
    print("1. 載入左影像蝦子資料...")
    left_image, left_masks, left_data = load_real_shrimp_data(left_json_path)
    if left_image is None:
        print("無法載入左影像，使用模擬資料...")
        demonstrate_simulated_sampling()
        return
    
    print("2. 載入右影像蝦子資料...")
    right_image, right_masks, right_data = load_real_shrimp_data(right_json_path)
    if right_image is None:
        print("無法載入右影像，使用模擬資料...")
        demonstrate_simulated_sampling()
        return
    
    print(f"左影像蝦子數量: {len(left_masks)}")
    print(f"右影像蝦子數量: {len(right_masks)}")
    
    if len(left_masks) == 0 or len(right_masks) == 0:
        print("沒有找到蝦子資料，使用模擬資料...")
        demonstrate_simulated_sampling()
        return
    
    # 3. 進行蝦子匹配
    print("\n3. 進行蝦子匹配...")
    matches = find_matching_shrimps(left_masks, right_masks, left_image, right_image)
    
    if len(matches) == 0:
        print("沒有找到匹配的蝦子對，使用第一隻蝦子進行示例...")
        # 如果沒有匹配，就使用第一隻蝦子
        left_mask = left_masks[0]
        right_mask = right_masks[0] if len(right_masks) > 0 else left_masks[0]
        ncc_score = 0.0
    else:
        # 使用匹配度最高的蝦子對
        best_match = max(matches, key=lambda x: x[2])
        left_mask, right_mask, ncc_score = best_match
        print(f"使用匹配度最高的蝦子對 (NCC = {ncc_score:.3f})")
    
    print("\n4. 提取蝦子輪廓點...")
    # 提取蝦子的像素點
    left_x, left_y = get_mask_points(left_mask)
    right_x, right_y = get_mask_points(right_mask)
    
    print(f"左蝦子像素點數: {len(left_x)}")
    print(f"右蝦子像素點數: {len(right_x)}")
    
    # 計算蝦子的範圍
    left_x_min, left_x_max = left_x.min(), left_x.max()
    left_y_min, left_y_max = left_y.min(), left_y.max()
    right_x_min, right_x_max = right_x.min(), right_x.max()
    right_y_min, right_y_max = right_y.min(), right_y.max()
    
    print(f"\n左蝦子範圍: x=[{left_x_min}, {left_x_max}], y=[{left_y_min}, {left_y_max}]")
    print(f"右蝦子範圍: x=[{right_x_min}, {right_x_max}], y=[{right_y_min}, {right_y_max}]")
    
    # 擬合二次曲線
    print("\n5. 擬合蝦子曲線...")
    left_params = fit_quadratic_curve(left_x, left_y)
    right_params = fit_quadratic_curve(right_x, right_y)
    
    print(f"左蝦子曲線參數: a={left_params[0]:.6f}, b={left_params[1]:.3f}, c={left_params[2]:.1f}")
    print(f"右蝦子曲線參數: a={right_params[0]:.6f}, b={right_params[1]:.3f}, c={right_params[2]:.1f}")
    print(f"蝦子匹配度 (NCC): {ncc_score:.3f}")
    
    # 設定採樣點數
    n_points = 5
    
    print(f"\n採樣點數: {n_points}")
    print("=" * 70)
    
    # 方法1：錯誤的傳統方法（固定x座標採樣）
    print("\n【方法1：傳統方法 - 錯誤】")
    print("在左右影像中使用相同的x座標範圍採樣")
    
    # 使用共同的x範圍
    common_x_min = max(left_x_min, right_x_min)
    common_x_max = min(left_x_max, right_x_max)
    
    if common_x_max <= common_x_min:
        print("警告: 左右蝦子沒有重疊的x範圍，使用左蝦子範圍")
        common_x_min, common_x_max = left_x_min, left_x_max
    
    traditional_x = np.linspace(common_x_min, common_x_max, n_points)
    
    # 計算對應的y值
    traditional_left_y = left_params[0] * traditional_x**2 + left_params[1] * traditional_x + left_params[2]
    traditional_right_y = right_params[0] * traditional_x**2 + right_params[1] * traditional_x + right_params[2]
    traditional_disparities = traditional_x - traditional_x  # 全部為0！
    
    print("採樣結果（傳統方法）：")
    for i, (x, ly, ry, d) in enumerate(zip(traditional_x, traditional_left_y, traditional_right_y, traditional_disparities)):
        print(f"  點{i+1}: x={x:6.1f} → 左y={ly:6.1f}, 右y={ry:6.1f}, 視差={d:6.1f}")
    
    print(f"平均視差: {np.mean(traditional_disparities):.1f} 像素 ← 錯誤！應該不為0")
    
    # 方法2：正確的相對位置方法
    print("\n【方法2：相對位置方法 - 正確】")  
    print("使用相對位置參數t在左右蝦子範圍內採樣")
    
    # 產生相對位置參數
    t_values = np.linspace(0, 1, n_points)
    
    # 映射到實際x座標
    relative_left_x = left_x_min + t_values * (left_x_max - left_x_min)
    relative_right_x = right_x_min + t_values * (right_x_max - right_x_min)
    
    # 計算對應的y座標
    relative_left_y = left_params[0] * relative_left_x**2 + left_params[1] * relative_left_x + left_params[2]
    relative_right_y = right_params[0] * relative_right_x**2 + right_params[1] * relative_right_x + right_params[2]
    
    # 計算視差
    relative_disparities = relative_left_x - relative_right_x
    
    print("採樣結果（相對位置方法）：")
    print("t值    左影像(x,y)      右影像(x,y)      視差    身體部位")
    print("-" * 65)
    body_parts = ["頭部", "前1/4", "中段", "後3/4", "尾部"]
    
    for i, (t, lx, ly, rx, ry, d) in enumerate(zip(t_values, relative_left_x, relative_left_y, 
                                                   relative_right_x, relative_right_y, relative_disparities)):
        part = body_parts[i] if i < len(body_parts) else f"第{i+1}點"
        print(f"{t:.2f}   ({lx:6.1f},{ly:6.1f})   ({rx:6.1f},{ry:6.1f})   {d:6.1f}   {part}")
    
    print(f"\n平均視差: {np.mean(relative_disparities):.1f} 像素 ← 正確！")
    print(f"視差標準差: {np.std(relative_disparities):.1f} 像素")
    
    # 創建視覺化圖表
    create_real_data_visualization(
        left_image, right_image,
        left_mask, right_mask,
        left_x, left_y, right_x, right_y,
        left_params, right_params,
        relative_left_x, relative_left_y,
        relative_right_x, relative_right_y,
        relative_disparities, ncc_score
    )

def demonstrate_simulated_sampling():
    """回退到模擬資料的示例"""
    print("\n=== 使用模擬資料進行示例 ===")
    
    # 模擬左右蝦子的範圍
    left_x_min, left_x_max = 700, 900    # 左蝦子x範圍
    right_x_min, right_x_max = 650, 850  # 右蝦子x範圍（整體向左偏移50像素）
    
    print(f"左蝦子x範圍: [{left_x_min}, {left_x_max}] (長度: {left_x_max - left_x_min})")
    print(f"右蝦子x範圍: [{right_x_min}, {right_x_max}] (長度: {right_x_max - right_x_min})")
    
    # 模擬蝦子的曲線方程（簡化為二次曲線）
    def left_curve(x):
        return -0.001 * (x - 800)**2 + 400
    
    def right_curve(x):
        return -0.001 * (x - 750)**2 + 410
    
    # 設定採樣點數
    n_points = 5
    
    print(f"\n採樣點數: {n_points}")
    print("=" * 50)
    
    # 相對位置方法
    print("\n【相對位置方法】")  
    print("使用相對位置參數t在左右蝦子範圍內採樣")
    
    # 產生相對位置參數
    t_values = np.linspace(0, 1, n_points)
    
    # 映射到實際x座標
    relative_left_x = left_x_min + t_values * (left_x_max - left_x_min)
    relative_right_x = right_x_min + t_values * (right_x_max - right_x_min)
    
    # 計算對應的y座標
    relative_left_y = left_curve(relative_left_x)
    relative_right_y = right_curve(relative_right_x)
    
    # 計算視差
    relative_disparities = relative_left_x - relative_right_x
    
    print("採樣結果（相對位置方法）：")
    print("t值    左影像(x,y)      右影像(x,y)      視差    身體部位")
    print("-" * 65)
    body_parts = ["頭部", "前1/4", "中段", "後3/4", "尾部"]
    
    for i, (t, lx, ly, rx, ry, d) in enumerate(zip(t_values, relative_left_x, relative_left_y, 
                                                   relative_right_x, relative_right_y, relative_disparities)):
        part = body_parts[i] if i < len(body_parts) else f"第{i+1}點"
        print(f"{t:.2f}   ({lx:6.1f},{ly:6.1f})   ({rx:6.1f},{ry:6.1f})   {d:6.1f}   {part}")
    
    print(f"\n平均視差: {np.mean(relative_disparities):.1f} 像素")
    print(f"視差標準差: {np.std(relative_disparities):.1f} 像素")

def demonstrate_relative_sampling():
    """演示相對位置採樣的概念"""
    
    print("=== 相對位置採樣示例 ===\n")
    
    # 嘗試使用真實資料，如果失敗則使用模擬資料
    try:
        demonstrate_real_data_sampling()
    except Exception as e:
        print(f"載入真實資料失敗: {e}")
        print("改用模擬資料進行示例...")
        demonstrate_simulated_sampling()

def create_real_data_visualization(left_image, right_image,
                                   left_mask, right_mask,
                                   left_x, left_y, right_x, right_y,
                                   left_params, right_params,
                                   sample_left_x, sample_left_y,
                                   sample_right_x, sample_right_y,
                                   disparities, ncc_score):
    """創建使用真實蝦子資料的視覺化圖表"""
    
    plt.figure(figsize=(20, 12))
    
    # 子圖1：左影像和蝦子遮罩
    plt.subplot(2, 3, 1)
    plt.imshow(left_image)
    
    # 繪製左蝦子遮罩輪廓 (統一使用紅色)
    contours, _ = cv2.findContours(left_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour = contour.reshape(-1, 2)
        plt.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, alpha=0.8)
    
    # 標出採樣點
    plt.scatter(sample_left_x, sample_left_y, c='red', s=100, zorder=5, marker='o', edgecolors='white', linewidth=2)
    
    # 添加點的編號
    for i, (x, y) in enumerate(zip(sample_left_x, sample_left_y)):
        plt.text(x+10, y-10, f'{i+1}', fontsize=12, color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8))
    
    plt.title(f'左影像 - 蝦子遮罩和採樣點\n(匹配度 NCC = {ncc_score:.3f})', fontsize=14)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # 子圖2：右影像和蝦子遮罩
    plt.subplot(2, 3, 2)
    plt.imshow(right_image)
    
    # 繪製右蝦子遮罩輪廓 (統一使用藍色)
    contours, _ = cv2.findContours(right_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour = contour.reshape(-1, 2)
        plt.plot(contour[:, 0], contour[:, 1], 'b-', linewidth=2, alpha=0.8)
    
    # 標出採樣點
    plt.scatter(sample_right_x, sample_right_y, c='blue', s=100, zorder=5, marker='s', edgecolors='white', linewidth=2)
    
    # 添加點的編號
    for i, (x, y) in enumerate(zip(sample_right_x, sample_right_y)):
        plt.text(x+10, y-10, f'{i+1}', fontsize=12, color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.8))
    
    plt.title('右影像 - 蝦子遮罩和採樣點', fontsize=14)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # 子圖3：蝦子曲線擬合
    plt.subplot(2, 3, 3)
    
    # 繪製原始點 (顛倒顏色: 左蝦子用紅色，右蝦子用藍色)
    plt.scatter(left_x, left_y, c='lightcoral', s=1, alpha=0.5, label='左蝦子像素')
    plt.scatter(right_x, right_y, c='lightblue', s=1, alpha=0.5, label='右蝦子像素')
    
    # 繪製擬合曲線
    x_range_left = np.linspace(left_x.min(), left_x.max(), 100)
    x_range_right = np.linspace(right_x.min(), right_x.max(), 100)
    
    y_fitted_left = left_params[0] * x_range_left**2 + left_params[1] * x_range_left + left_params[2]
    y_fitted_right = right_params[0] * x_range_right**2 + right_params[1] * x_range_right + right_params[2]
    
    # 顛倒曲線顏色: 左蝦子用紅色，右蝦子用藍色
    plt.plot(x_range_left, y_fitted_left, 'r-', linewidth=3, label='左蝦子曲線')
    plt.plot(x_range_right, y_fitted_right, 'b-', linewidth=3, label='右蝦子曲線')
    
    # 標出採樣點 (顛倒顏色: 左蝦子用紅色，右蝦子用藍色)
    plt.scatter(sample_left_x, sample_left_y, c='red', s=150, zorder=5, marker='o', edgecolors='white', linewidth=2)
    plt.scatter(sample_right_x, sample_right_y, c='blue', s=150, zorder=5, marker='s', edgecolors='white', linewidth=2)
    
    # 繪製對應關係
    for i, (lx, ly, rx, ry) in enumerate(zip(sample_left_x, sample_left_y, sample_right_x, sample_right_y)):
        plt.plot([lx, rx], [ly, ry], 'g--', alpha=0.7, linewidth=2)
        plt.text((lx + rx)/2, (ly + ry)/2 + 5, f'{i+1}', 
                fontsize=10, ha='center', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    
    plt.xlabel('X 座標 (像素)', fontsize=12)
    plt.ylabel('Y 座標 (像素)', fontsize=12)
    plt.title('蝦子曲線擬合和採樣點對應關係', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 動態設定X軸和Y軸範圍，保持1920:1080比例的同時盡可能放大
    # 計算所有點的範圍
    all_x = np.concatenate([left_x, right_x, sample_left_x, sample_right_x])
    all_y = np.concatenate([left_y, right_y, sample_left_y, sample_right_y])
    
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    
    # 添加邊距
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    x_range = (x_max - x_min) + 2 * x_margin
    y_range = (y_max - y_min) + 2 * y_margin
    
    # 計算1920:1080的比例
    aspect_ratio = 1920 / 1080
    
    # 調整範圍以保持比例
    if x_range / y_range > aspect_ratio:
        # X範圍太大，調整Y範圍
        target_y_range = x_range / aspect_ratio
        extra_y = (target_y_range - y_range) / 2
        y_min_adj = y_min - y_margin - extra_y
        y_max_adj = y_max + y_margin + extra_y
        x_min_adj = x_min - x_margin
        x_max_adj = x_max + x_margin
    else:
        # Y範圍太大，調整X範圍
        target_x_range = y_range * aspect_ratio
        extra_x = (target_x_range - x_range) / 2
        x_min_adj = x_min - x_margin - extra_x
        x_max_adj = x_max + x_margin + extra_x
        y_min_adj = y_min - y_margin
        y_max_adj = y_max + y_margin
    
    plt.xlim(x_min_adj, x_max_adj)
    plt.ylim(y_min_adj, y_max_adj)
    # 翻轉Y軸，使其與影像座標系統一致（Y軸向下為正）
    plt.gca().invert_yaxis()
    
    # 子圖4：視差分布
    plt.subplot(2, 3, 4)
    point_indices = range(1, len(disparities) + 1)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    plt.bar(point_indices, disparities, color=colors[:len(disparities)])
    plt.xlabel('採樣點編號', fontsize=12)
    plt.ylabel('視差 (像素)', fontsize=12)
    plt.title('各採樣點的視差分布', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 添加數值標籤
    for i, d in enumerate(disparities):
        plt.text(i+1, d + (max(disparities) - min(disparities)) * 0.02, f'{d:.1f}', 
                ha='center', va='bottom', fontsize=11, weight='bold')
    
    # 子圖5：x座標比較
    plt.subplot(2, 3, 5)
    plt.plot(point_indices, sample_left_x, 'ro-', label='左影像 x座標', linewidth=3, markersize=10)
    plt.plot(point_indices, sample_right_x, 'bo-', label='右影像 x座標', linewidth=3, markersize=10)
    plt.xlabel('採樣點編號', fontsize=12)
    plt.ylabel('X 座標 (像素)', fontsize=12)
    plt.title('左右影像 X 座標比較', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子圖6：y座標比較
    plt.subplot(2, 3, 6)
    plt.plot(point_indices, sample_left_y, 'ro-', label='左影像 y座標', linewidth=3, markersize=10)
    plt.plot(point_indices, sample_right_y, 'bo-', label='右影像 y座標', linewidth=3, markersize=10)
    plt.xlabel('採樣點編號', fontsize=12)
    plt.ylabel('Y 座標 (像素)', fontsize=12)
    plt.title('左右影像 Y 座標比較', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 翻轉Y軸，使其與影像座標系統一致
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    output_filename = '真實蝦子資料相對位置採樣示例.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n真實資料視覺化圖表已儲存為: {output_filename}")

def create_visualization(left_x_min, left_x_max, right_x_min, right_x_max,
                        left_curve, right_curve,
                        sample_left_x, sample_left_y,
                        sample_right_x, sample_right_y,
                        disparities):
    """創建視覺化圖表"""
    
    plt.figure(figsize=(15, 10))
    
    # 子圖1：左右蝦子曲線
    plt.subplot(2, 2, 1)
    
    # 繪製完整的蝦子曲線
    x_range = np.linspace(600, 950, 200)
    left_y_range = left_curve(x_range)
    right_y_range = right_curve(x_range)
    
    # 顛倒顏色: 左蝦子用紅色，右蝦子用藍色
    plt.plot(x_range, left_y_range, 'r-', linewidth=2, label='左蝦子曲線', alpha=0.3)
    plt.plot(x_range, right_y_range, 'b-', linewidth=2, label='右蝦子曲線', alpha=0.3)
    
    # 標出蝦子的有效範圍
    left_valid_x = np.linspace(left_x_min, left_x_max, 50)
    right_valid_x = np.linspace(right_x_min, right_x_max, 50)
    plt.plot(left_valid_x, left_curve(left_valid_x), 'r-', linewidth=4, label='左蝦子')
    plt.plot(right_valid_x, right_curve(right_valid_x), 'b-', linewidth=4, label='右蝦子')
    
    # 標出採樣點 (顛倒顏色: 左蝦子用紅色，右蝦子用藍色)
    plt.scatter(sample_left_x, sample_left_y, c='red', s=100, zorder=5, marker='o')
    plt.scatter(sample_right_x, sample_right_y, c='blue', s=100, zorder=5, marker='s')
    
    # 繪製對應關係
    for i, (lx, ly, rx, ry) in enumerate(zip(sample_left_x, sample_left_y, sample_right_x, sample_right_y)):
        plt.plot([lx, rx], [ly, ry], 'g--', alpha=0.7, linewidth=1)
        plt.text((lx + rx)/2, (ly + ry)/2 + 10, f'{i+1}', 
                fontsize=10, ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.xlabel('X 座標 (像素)')
    plt.ylabel('Y 座標 (像素)')
    plt.title('蝦子曲線和採樣點對應關係')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 動態設定X軸和Y軸範圍，保持1920:1080比例的同時盡可能放大
    # 計算所有點的範圍
    all_x = np.concatenate([sample_left_x, sample_right_x])
    all_y = np.concatenate([sample_left_y, sample_right_y])
    
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    
    # 添加邊距
    x_margin = max((x_max - x_min) * 0.2, 50)  # 至少50像素邊距
    y_margin = max((y_max - y_min) * 0.2, 50)  # 至少50像素邊距
    
    x_range = (x_max - x_min) + 2 * x_margin
    y_range = (y_max - y_min) + 2 * y_margin
    
    # 計算1920:1080的比例
    aspect_ratio = 1920 / 1080
    
    # 調整範圍以保持比例
    if x_range / y_range > aspect_ratio:
        # X範圍太大，調整Y範圍
        target_y_range = x_range / aspect_ratio
        extra_y = (target_y_range - y_range) / 2
        y_min_adj = y_min - y_margin - extra_y
        y_max_adj = y_max + y_margin + extra_y
        x_min_adj = x_min - x_margin
        x_max_adj = x_max + x_margin
    else:
        # Y範圍太大，調整X範圍
        target_x_range = y_range * aspect_ratio
        extra_x = (target_x_range - x_range) / 2
        x_min_adj = x_min - x_margin - extra_x
        x_max_adj = x_max + x_margin + extra_x
        y_min_adj = y_min - y_margin
        y_max_adj = y_max + y_margin
    
    plt.xlim(x_min_adj, x_max_adj)
    plt.ylim(y_min_adj, y_max_adj)
    # 翻轉Y軸，使其與影像座標系統一致（Y軸向下為正）
    plt.gca().invert_yaxis()
    
    # 子圖2：視差分布
    plt.subplot(2, 2, 2)
    point_indices = range(1, len(disparities) + 1)
    plt.bar(point_indices, disparities, color=['blue', 'green', 'orange', 'purple', 'red'])
    plt.xlabel('採樣點編號')
    plt.ylabel('視差 (像素)')
    plt.title('各採樣點的視差分布')
    plt.grid(True, alpha=0.3)
    
    # 添加數值標籤
    for i, d in enumerate(disparities):
        plt.text(i+1, d+1, f'{d:.1f}', ha='center', va='bottom')
    
    # 子圖3：x座標比較
    plt.subplot(2, 2, 3)
    plt.plot(point_indices, sample_left_x, 'ro-', label='左影像 x座標', linewidth=2, markersize=8)
    plt.plot(point_indices, sample_right_x, 'bo-', label='右影像 x座標', linewidth=2, markersize=8)
    plt.xlabel('採樣點編號')
    plt.ylabel('X 座標 (像素)')
    plt.title('左右影像 X 座標比較')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子圖4：y座標比較
    plt.subplot(2, 2, 4)
    plt.plot(point_indices, sample_left_y, 'ro-', label='左影像 y座標', linewidth=2, markersize=8)
    plt.plot(point_indices, sample_right_y, 'bo-', label='右影像 y座標', linewidth=2, markersize=8)
    plt.xlabel('採樣點編號')
    plt.ylabel('Y 座標 (像素)')
    plt.title('左右影像 Y 座標比較')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # 翻轉Y軸，使其與影像座標系統一致
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('相對位置採樣示例.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n視覺化圖表已儲存為: 相對位置採樣示例.png")

if __name__ == "__main__":
    demonstrate_relative_sampling()
