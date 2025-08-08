#!/usr/bin/env python3
"""
分析 SAM2 JSON 檔案並對掩碼計算最佳迴歸曲線
功能:
1. 讀取 samples 資料夾下的 JSON 檔案
2. 解碼 RLE 掩碼
3. 計算最小距離和的迴歸曲線 (線性、二次、三次)
4. 繪製原始圖片、bounding box 和迴歸曲線
"""

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pycocotools import mask as mask_utils
import os
import glob
from pathlib import Path
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

# 解決中文顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

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

def total_distance_to_curve(params, x_points, y_points, curve_type='linear'):
    """
    計算所有點到曲線的總距離
    
    Args:
        params (array): 曲線參數
        x_points (array): 點的 x 座標
        y_points (array): 點的 y 座標
        curve_type (str): 曲線類型 ('linear', 'quadratic', 'cubic')
        
    Returns:
        float: 總距離
    """
    if curve_type == 'linear':
        # y = ax + b
        a, b = params
        # 點到直線 ax - y + b = 0 的距離
        distances = np.abs(a * x_points - y_points + b) / np.sqrt(a**2 + 1)
    
    elif curve_type == 'quadratic':
        # y = ax^2 + bx + c
        a, b, c = params
        # 對於二次曲線，使用最近點距離（較複雜，這裡使用垂直距離作為近似）
        curve_y = a * x_points**2 + b * x_points + c
        distances = np.abs(y_points - curve_y)
    
    elif curve_type == 'cubic':
        # y = ax^3 + bx^2 + cx + d
        a, b, c, d = params
        curve_y = a * x_points**3 + b * x_points**2 + c * x_points + d
        distances = np.abs(y_points - curve_y)
    
    return np.sum(distances)

def fit_optimal_curve(x_points, y_points, curve_type='linear'):
    """
    擬合最佳迴歸曲線，最小化總距離
    
    Args:
        x_points (array): 點的 x 座標
        y_points (array): 點的 y 座標
        curve_type (str): 曲線類型
        
    Returns:
        array: 最佳曲線參數
    """
    # 正規化座標以改善數值穩定性
    x_min, x_max = x_points.min(), x_points.max()
    y_min, y_max = y_points.min(), y_points.max()
    
    x_norm = (x_points - x_min) / (x_max - x_min) if x_max != x_min else x_points
    y_norm = (y_points - y_min) / (y_max - y_min) if y_max != y_min else y_points
    
    # 使用最小二乘法作為初始猜測
    if curve_type == 'linear':
        poly_features = PolynomialFeatures(degree=1, include_bias=True)
        initial_guess = [1.0, 0.0]
    elif curve_type == 'quadratic':
        poly_features = PolynomialFeatures(degree=2, include_bias=True)
        initial_guess = [0.0, 1.0, 0.0]
    elif curve_type == 'cubic':
        poly_features = PolynomialFeatures(degree=3, include_bias=True)
        initial_guess = [0.0, 0.0, 1.0, 0.0]
    
    # 使用最小二乘法獲得初始估計
    try:
        X_poly = poly_features.fit_transform(x_norm.reshape(-1, 1))
        model = LinearRegression(fit_intercept=False)
        model.fit(X_poly, y_norm)
        
        if curve_type == 'linear':
            initial_guess = [model.coef_[1], model.coef_[0]]
        elif curve_type == 'quadratic':
            initial_guess = [model.coef_[2], model.coef_[1], model.coef_[0]]
        elif curve_type == 'cubic':
            initial_guess = [model.coef_[3], model.coef_[2], model.coef_[1], model.coef_[0]]
    except:
        pass  # 使用默認初始猜測
    
    # 最小化總距離
    result = minimize(
        total_distance_to_curve,
        initial_guess,
        args=(x_norm, y_norm, curve_type),
        method='BFGS'
    )
    
    # 將參數轉換回原始座標系
    params_norm = result.x
    
    if curve_type == 'linear':
        a_norm, b_norm = params_norm
        # y_norm = a_norm * x_norm + b_norm
        # (y - y_min)/(y_max - y_min) = a_norm * (x - x_min)/(x_max - x_min) + b_norm
        # y = a_norm * (y_max - y_min)/(x_max - x_min) * (x - x_min) + b_norm * (y_max - y_min) + y_min
        if x_max != x_min and y_max != y_min:
            a = a_norm * (y_max - y_min) / (x_max - x_min)
            b = b_norm * (y_max - y_min) + y_min - a * x_min
        else:
            a, b = 0, np.mean(y_points)
        params = [a, b]
        
    elif curve_type == 'quadratic':
        a_norm, b_norm, c_norm = params_norm
        if x_max != x_min and y_max != y_min:
            scale_x = (x_max - x_min)
            scale_y = (y_max - y_min)
            a = a_norm * scale_y / (scale_x**2)
            b = b_norm * scale_y / scale_x - 2 * a * x_min
            c = c_norm * scale_y + y_min - a * x_min**2 - b * x_min
        else:
            a, b, c = 0, 0, np.mean(y_points)
        params = [a, b, c]
        
    elif curve_type == 'cubic':
        a_norm, b_norm, c_norm, d_norm = params_norm
        if x_max != x_min and y_max != y_min:
            scale_x = (x_max - x_min)
            scale_y = (y_max - y_min)
            a = a_norm * scale_y / (scale_x**3)
            b = b_norm * scale_y / (scale_x**2) - 3 * a * x_min
            c = c_norm * scale_y / scale_x - 3 * a * x_min**2 - 2 * b * x_min
            d = d_norm * scale_y + y_min - a * x_min**3 - b * x_min**2 - c * x_min
        else:
            a, b, c, d = 0, 0, 0, np.mean(y_points)
        params = [a, b, c, d]
    
    return np.array(params)

def generate_curve_points(params, x_range, curve_type='linear', bbox=None):
    """
    根據參數生成曲線上的點，限制在 bounding box 內
    
    Args:
        params (array): 曲線參數
        x_range (tuple): x 座標範圍 (x_min, x_max)
        curve_type (str): 曲線類型
        bbox (list): bounding box [x1, y1, x2, y2] (xyxy 格式)
        
    Returns:
        tuple: (x_curve, y_curve) 曲線上的點
    """
    x_min, x_max = x_range
    
    # 如果有 bbox，限制 x 範圍
    if bbox is not None:
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        x_min = max(x_min, bbox_x1)
        x_max = min(x_max, bbox_x2)
    
    x_curve = np.linspace(x_min, x_max, 100)
    
    if curve_type == 'linear':
        a, b = params
        y_curve = a * x_curve + b
    elif curve_type == 'quadratic':
        a, b, c = params
        y_curve = a * x_curve**2 + b * x_curve + c
    elif curve_type == 'cubic':
        a, b, c, d = params
        y_curve = a * x_curve**3 + b * x_curve**2 + c * x_curve + d
    
    # 如果有 bbox，限制 y 範圍
    if bbox is not None:
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        # 只保留在 bbox 內的點
        valid_mask = (y_curve >= bbox_y1) & (y_curve <= bbox_y2)
        x_curve = x_curve[valid_mask]
        y_curve = y_curve[valid_mask]
    
    return x_curve, y_curve

def process_json_file(json_path):
    """
    處理單個 JSON 檔案
    
    Args:
        json_path (str): JSON 檔案路徑
        
    Returns:
        dict: 處理結果
    """
    print(f"\n處理檔案: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 獲取圖片路徑
    image_path = data['image_path']
    # 轉換路徑分隔符
    image_path = image_path.replace('\\', os.sep)
    
    # 構建完整的圖片路徑
    json_dir = os.path.dirname(json_path)
    # 找到專案根目錄 (包含 extracted_frames_left 的目錄)
    project_root = json_dir
    while not os.path.exists(os.path.join(project_root, 'extracted_frames_left')) and project_root != os.path.dirname(project_root):
        project_root = os.path.dirname(project_root)
    
    full_image_path = os.path.join(project_root, image_path)
    
    print(f"圖片路徑: {full_image_path}")
    
    # 檢查圖片是否存在
    if not os.path.exists(full_image_path):
        print(f"警告: 圖片檔案不存在: {full_image_path}")
        return None
    
    # 讀取圖片
    image = cv2.imread(full_image_path)
    if image is None:
        print(f"錯誤: 無法讀取圖片: {full_image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = {
        'json_path': json_path,
        'image_path': full_image_path,
        'image': image_rgb,
        'annotations': []
    }
    
    # 處理每個 annotation
    for i, annotation in enumerate(data['annotations']):
        print(f"  處理第 {i+1} 個 {annotation['class_name']} (信心度: {annotation['score'][0]:.3f})")
        
        # 解碼掩碼
        mask = decode_rle_to_mask(annotation['segmentation'])
        
        # 獲取掩碼內的點
        x_points, y_points = get_mask_points(mask)
        
        if len(x_points) == 0:
            print(f"    警告: 掩碼為空")
            continue
        
        # 處理 bounding box - 檢查格式
        bbox = annotation['bbox']
        if 'box_format' in data and data['box_format'] == 'xyxy':
            # xyxy 格式: [x1, y1, x2, y2]
            bbox_xyxy = bbox
        else:
            # 假設是 xywh 格式: [x, y, width, height]
            x, y, w, h = bbox
            bbox_xyxy = [x, y, x + w, y + h]
        
        # 計算不同類型的迴歸曲線
        curve_results = {}
        for curve_type in ['linear', 'quadratic', 'cubic']:
            try:
                params = fit_optimal_curve(x_points, y_points, curve_type)
                total_dist = total_distance_to_curve(params, x_points, y_points, curve_type)
                
                curve_results[curve_type] = {
                    'params': params,
                    'total_distance': total_dist,
                    'avg_distance': total_dist / len(x_points)
                }
                
                print(f"    {curve_type.capitalize()}: 總距離={total_dist:.2f}, 平均距離={total_dist/len(x_points):.3f}")
                
            except Exception as e:
                print(f"    {curve_type.capitalize()} 擬合失敗: {e}")
                curve_results[curve_type] = None
        
        # 找出最佳曲線（總距離最小）
        best_curve_type = None
        best_distance = float('inf')
        
        for curve_type, result in curve_results.items():
            if result and result['total_distance'] < best_distance:
                best_distance = result['total_distance']
                best_curve_type = curve_type
        
        annotation_result = {
            'class_name': annotation['class_name'],
            'bbox': bbox,
            'bbox_xyxy': bbox_xyxy,
            'score': annotation['score'][0],
            'mask': mask,
            'mask_points': (x_points, y_points),
            'curve_results': curve_results,
            'best_curve_type': best_curve_type
        }
        
        if best_curve_type:
            print(f"    最佳曲線: {best_curve_type.capitalize()}")
        
        results['annotations'].append(annotation_result)
    
    return results

def visualize_results(results, output_dir=None):
    """
    視覺化結果 - 在原始圖片上顯示最佳迴歸曲線
    
    Args:
        results (dict): 處理結果
        output_dir (str): 輸出目錄，如果為 None 則顯示圖片
    """
    image = results['image']
    annotations = results['annotations']
    
    if len(annotations) == 0:
        return
    
    # 創建圖片副本
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(image)
    
    # 顏色映射
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    for i, annotation in enumerate(annotations):
        color = colors[i % len(colors)]
        
        # 繪製 bounding box
        bbox_xyxy = annotation['bbox_xyxy']
        x1, y1, x2, y2 = bbox_xyxy
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none', linestyle='--', alpha=0.7)
        ax.add_patch(rect)
        
        # 不顯示掩碼點，只顯示迴歸曲線
        # x_points, y_points = annotation['mask_points']
        # ax.scatter(x_sample, y_sample, c=color, s=0.5, alpha=0.3, label=f'{annotation["class_name"]} 掩碼點')
        
        # 繪製最佳迴歸曲線
        best_type = annotation['best_curve_type']
        if best_type and annotation['curve_results'][best_type]:
            params = annotation['curve_results'][best_type]['params']
            x_points, y_points = annotation['mask_points']
            x_range = (x_points.min(), x_points.max())
            
            # 生成曲線點，限制在 bounding box 內
            x_curve, y_curve = generate_curve_points(params, x_range, best_type, bbox_xyxy)
            
            if len(x_curve) > 0:  # 確保有有效的曲線點
                ax.plot(x_curve, y_curve, color=color, linewidth=3, 
                       label=f'{annotation["class_name"]} {best_type} curve (avg dist: {annotation["curve_results"][best_type]["avg_distance"]:.2f})')
        
        # 添加類別標籤
        ax.text(x1, y1-5, f'{annotation["class_name"]} ({annotation["score"]:.2f})', 
               color=color, fontsize=12, fontweight='bold', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_title('Object Detection with Best Regression Curves', fontsize=16, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(0.02, 0.98), loc='upper left')
    ax.axis('off')
    
    plt.tight_layout()
    
    # 保存或顯示圖片
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_filename = os.path.basename(results['json_path'])
        output_filename = f"best_regression_{json_filename.replace('.json', '.png')}"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"結果已保存到: {output_path}")
        plt.close()
    else:
        plt.show()

def main():
    """
    主函數
    """
    # 尋找所有 JSON 檔案
    samples_dir = r"d:\Git\Grounded-SAM-2\samples"
    json_files = glob.glob(os.path.join(samples_dir, "**", "*.json"), recursive=True)
    
    if not json_files:
        print("在 samples 資料夾中找不到 JSON 檔案")
        return
    
    print(f"找到 {len(json_files)} 個 JSON 檔案")
    
    # 創建輸出目錄
    output_dir = os.path.join(os.path.dirname(__file__), "regression_analysis_results")
    
    # 處理每個 JSON 檔案
    for json_file in json_files:
        try:
            results = process_json_file(json_file)
            if results and results['annotations']:
                visualize_results(results, output_dir)
            else:
                print(f"跳過檔案 {json_file} (無有效註釋)")
        except Exception as e:
            print(f"處理檔案 {json_file} 時發生錯誤: {e}")
            continue
    
    print(f"\n所有結果已保存到: {output_dir}")

if __name__ == "__main__":
    main()
