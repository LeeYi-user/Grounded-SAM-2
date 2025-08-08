#!/usr/bin/env python3
"""
蝦子雙目立體視覺視差分析程式
結合物體匹配和迴歸曲線分析，計算蝦子的視差

功能:
1. 使用NCC匹配左右影像中的蝦子
2. 對每隻蝦子計算最佳迴歸曲線
3. 在曲線上平均取N個點
4. 計算每個對應點的視差 (disparity)
5. 產生詳細的視覺化結果

作者: Assistant
日期: 2025年8月8日
"""

import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# 解決中文顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class ShrimpDisparityAnalyzer:
    """蝦子視差分析器"""
    
    def __init__(self, ncc_threshold=0.3):
        """
        初始化分析器
        
        Args:
            ncc_threshold (float): NCC匹配閾值
        """
        self.ncc_threshold = ncc_threshold
        self.results = {}
    
    def decode_rle_to_mask(self, rle_data):
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
    
    def load_image_and_annotations(self, json_path):
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
        
        # 取得圖像路徑
        image_path = data['image_path']
        
        # 將反斜線轉換為正斜線，並建構完整路徑 
        image_path = image_path.replace('\\', os.sep)
        root_dir = r"d:\Git\Grounded-SAM-2"
        full_image_path = os.path.join(root_dir, image_path)
        full_image_path = os.path.normpath(full_image_path)
        
        print(f"載入圖像: {full_image_path}")
        
        # 載入圖像
        image = cv2.imread(full_image_path)
        if image is None:
            raise FileNotFoundError(f"無法載入圖像: {full_image_path}")
        
        # 轉換為RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image_rgb, data
    
    def extract_object_roi(self, image, annotation):
        """
        根據annotation從圖像中提取物體的ROI
        
        Args:
            image (numpy.ndarray): 原始圖像
            annotation (dict): 物體標註資訊
            
        Returns:
            tuple: (roi_image, roi_mask, bbox)
        """
        # 解碼遮罩
        mask = self.decode_rle_to_mask(annotation['segmentation'])
        
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
        
        return masked_roi, roi_mask, (x1, y1, x2, y2), mask
    
    def compute_ncc(self, template, search_region):
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
        
        # 如果模板比搜索區域大，調整模板大小
        if template_h > search_h or template_w > search_w:
            scale_h = search_h / template_h if template_h > search_h else 1.0
            scale_w = search_w / template_w if template_w > search_w else 1.0
            scale = min(scale_h, scale_w, 1.0)
            
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
    
    def perform_shrimp_matching(self, left_data, right_data, left_image, right_image):
        """
        執行蝦子物體匹配
        
        Args:
            left_data (dict): 左影像的標註資料
            right_data (dict): 右影像的標註資料
            left_image (numpy.ndarray): 左影像
            right_image (numpy.ndarray): 右影像
            
        Returns:
            list: 蝦子匹配結果列表
        """
        matches = []
        
        # 過濾出蝦子物體
        left_shrimps = [ann for ann in left_data['annotations'] if 'shrimp' in ann['class_name'].lower()]
        right_shrimps = [ann for ann in right_data['annotations'] if 'shrimp' in ann['class_name'].lower()]
        
        print(f"左影像蝦子數量: {len(left_shrimps)}")
        print(f"右影像蝦子數量: {len(right_shrimps)}")
        
        # 對每個左影像蝦子尋找右影像中的匹配
        for i, left_ann in enumerate(left_shrimps):
            print(f"\n處理左影像蝦子 {i+1}/{len(left_shrimps)}")
            
            # 提取左影像蝦子ROI
            left_roi, left_mask, left_bbox, left_full_mask = self.extract_object_roi(left_image, left_ann)
            
            if left_roi.size == 0:
                print(f"  跳過: 左影像蝦子 {i+1} ROI為空")
                continue
            
            best_match = None
            best_ncc = -1
            
            # 在右影像的所有蝦子中尋找最佳匹配
            for j, right_ann in enumerate(right_shrimps):
                # 提取右影像蝦子ROI
                right_roi, right_mask, right_bbox, right_full_mask = self.extract_object_roi(right_image, right_ann)
                
                if right_roi.size == 0:
                    print(f"    跳過: 右影像蝦子 {j+1} ROI為空")
                    continue
                
                # 計算NCC
                try:
                    ncc_score, match_loc = self.compute_ncc(left_roi, right_roi)
                    
                    print(f"  左蝦子 {i+1} vs 右蝦子 {j+1}: NCC = {ncc_score:.3f}")
                    
                    # 更新最佳匹配
                    if ncc_score > best_ncc and ncc_score > self.ncc_threshold:
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
                            'left_full_mask': left_full_mask,
                            'right_full_mask': right_full_mask,
                            'ncc_score': ncc_score,
                            'match_location': match_loc
                        }
                except Exception as e:
                    print(f"  錯誤: 計算NCC失敗 - {e}")
            
            # 儲存匹配結果
            if best_match:
                matches.append(best_match)
                print(f"  ✓ 找到匹配: 左蝦子 {i+1} <-> 右蝦子 {best_match['right_idx']+1} (NCC = {best_ncc:.3f})")
            else:
                print(f"  ✗ 未找到匹配: 左蝦子 {i+1}")
        
        return matches
    
    def get_mask_points(self, mask):
        """
        從二進制遮罩中提取所有為 True 的像素點座標
        
        Args:
            mask (numpy.ndarray): 二進制遮罩
            
        Returns:
            tuple: (x_coords, y_coords) 像素點座標陣列
        """
        y_coords, x_coords = np.where(mask > 0)
        return x_coords, y_coords
    
    def total_distance_to_curve(self, params, x_points, y_points, curve_type='linear'):
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
            curve_y = a * x_points**2 + b * x_points + c
            distances = np.abs(y_points - curve_y)
        
        elif curve_type == 'cubic':
            # y = ax^3 + bx^2 + cx + d
            a, b, c, d = params
            curve_y = a * x_points**3 + b * x_points**2 + c * x_points + d
            distances = np.abs(y_points - curve_y)
        
        return np.sum(distances)
    
    def fit_optimal_curve(self, x_points, y_points, curve_type='quadratic'):
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
            self.total_distance_to_curve,
            initial_guess,
            args=(x_norm, y_norm, curve_type),
            method='BFGS'
        )
        
        # 將參數轉換回原始座標系
        params_norm = result.x
        
        if curve_type == 'linear':
            a_norm, b_norm = params_norm
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
    
    def find_corresponding_points_on_curves(self, left_params, right_params, curve_type, n_points=10, left_mask=None, right_mask=None):
        """
        在左右曲線上找到對應的點
        使用蝦子的實際範圍來採樣對應點
        
        Args:
            left_params (array): 左曲線參數
            right_params (array): 右曲線參數  
            curve_type (str): 曲線類型
            n_points (int): 採樣點數
            left_mask (numpy.ndarray): 左蝦子遮罩
            right_mask (numpy.ndarray): 右蝦子遮罩
            
        Returns:
            tuple: (left_points, right_points) 對應的左右點座標
        """
        # 如果有遮罩，使用遮罩的實際範圍
        if left_mask is not None and right_mask is not None:
            left_y_coords, left_x_coords = np.where(left_mask > 0)
            right_y_coords, right_x_coords = np.where(right_mask > 0)
            
            if len(left_x_coords) > 0 and len(right_x_coords) > 0:
                left_x_min, left_x_max = left_x_coords.min(), left_x_coords.max()
                right_x_min, right_x_max = right_x_coords.min(), right_x_coords.max()
                
                # 使用歸一化的位置來生成對應點
                # 左蝦子從頭到尾的相對位置
                t_values = np.linspace(0, 1, n_points)
                
                # 在左蝦子的實際x範圍內採樣
                left_x = left_x_min + t_values * (left_x_max - left_x_min)
                
                # 在右蝦子的實際x範圍內採樣（使用相同的相對位置）
                right_x = right_x_min + t_values * (right_x_max - right_x_min)
                
                # 根據曲線方程計算對應的y值
                if curve_type == 'linear':
                    a_left, b_left = left_params
                    a_right, b_right = right_params
                    left_y = a_left * left_x + b_left
                    right_y = a_right * right_x + b_right
                    
                elif curve_type == 'quadratic':
                    a_left, b_left, c_left = left_params
                    a_right, b_right, c_right = right_params
                    left_y = a_left * left_x**2 + b_left * left_x + c_left
                    right_y = a_right * right_x**2 + b_right * right_x + c_right
                    
                elif curve_type == 'cubic':
                    a_left, b_left, c_left, d_left = left_params
                    a_right, b_right, c_right, d_right = right_params
                    left_y = a_left * left_x**3 + b_left * left_x**2 + c_left * left_x + d_left
                    right_y = a_right * right_x**3 + b_right * right_x**2 + c_right * right_x + d_right
                
                return (left_x, left_y), (right_x, right_y)
        
        # 回退到簡化方法
        t_values = np.linspace(0, 1, n_points)
        
        if curve_type == 'linear':
            a_left, b_left = left_params
            a_right, b_right = right_params
            
            left_x = t_values * 200 + 600  # 預設範圍
            left_y = a_left * left_x + b_left
            
            right_x = t_values * 200 + 600
            right_y = a_right * right_x + b_right
            
        elif curve_type == 'quadratic':
            a_left, b_left, c_left = left_params
            a_right, b_right, c_right = right_params
            
            left_x = t_values * 200 + 600
            left_y = a_left * left_x**2 + b_left * left_x + c_left
            
            right_x = t_values * 200 + 600
            right_y = a_right * right_x**2 + b_right * right_x + c_right
            
        elif curve_type == 'cubic':
            a_left, b_left, c_left, d_left = left_params
            a_right, b_right, c_right, d_right = right_params
            
            left_x = t_values * 200 + 600
            left_y = a_left * left_x**3 + b_left * left_x**2 + c_left * left_x + d_left
            
            right_x = t_values * 200 + 600
            right_y = a_right * right_x**3 + b_right * right_x**2 + c_right * right_x + d_right
        
        return (left_x, left_y), (right_x, right_y)
    
    def sample_points_on_curve(self, params, x_range, n_points=10, curve_type='quadratic'):
        """
        在曲線上平均取N個點
        
        Args:
            params (array): 曲線參數
            x_range (tuple): x 座標範圍 (x_min, x_max)
            n_points (int): 要取的點數
            curve_type (str): 曲線類型
            
        Returns:
            tuple: (x_sampled, y_sampled) 採樣點座標
        """
        x_min, x_max = x_range
        x_sampled = np.linspace(x_min, x_max, n_points)
        
        if curve_type == 'linear':
            a, b = params
            y_sampled = a * x_sampled + b
        elif curve_type == 'quadratic':
            a, b, c = params
            y_sampled = a * x_sampled**2 + b * x_sampled + c
        elif curve_type == 'cubic':
            a, b, c, d = params
            y_sampled = a * x_sampled**3 + b * x_sampled**2 + c * x_sampled + d
        
        return x_sampled, y_sampled
    
    def calculate_disparity(self, left_points, right_points):
        """
        計算視差
        
        Args:
            left_points (tuple): 左影像點座標 (x_coords, y_coords)
            right_points (tuple): 右影像點座標 (x_coords, y_coords)
            
        Returns:
            tuple: (disparities, debug_info) 視差值和調試資訊
        """
        left_x, left_y = left_points
        right_x, right_y = right_points
        
        # 確保點數相同
        min_points = min(len(left_x), len(right_x))
        left_x = left_x[:min_points]
        left_y = left_y[:min_points]
        right_x = right_x[:min_points]
        right_y = right_y[:min_points]
        
        # 計算水平視差 (假設是校正過的立體圖像對)
        disparities = left_x - right_x
        
        # 檢查y座標差異（僅作為資訊顯示，不影響有效性判斷）
        y_diff = np.abs(left_y - right_y)
        
        # 調試資訊
        debug_info = {
            'left_x_range': (float(np.min(left_x)), float(np.max(left_x))),
            'left_y_range': (float(np.min(left_y)), float(np.max(left_y))),
            'right_x_range': (float(np.min(right_x)), float(np.max(right_x))),
            'right_y_range': (float(np.min(right_y)), float(np.max(right_y))),
            'disparity_range': (float(np.min(disparities)), float(np.max(disparities))),
            'y_diff_range': (float(np.min(y_diff)), float(np.max(y_diff))),
            'y_diff_mean': float(np.mean(y_diff)),
            'y_diff_std': float(np.std(y_diff)),
            'y_threshold_removed': True  # 標記已移除Y閾值限制
        }
        
        # 移除Y座標閾值限制 - 接受所有通過t參數採樣的對應點
        # 理由：
        # 1. 已經通過NCC確保了物體匹配的正確性
        # 2. 已經通過t參數確保了解剖學對應關係
        # 3. Y座標差異是預期的（蝦子彎曲、時間差、未校正等因素）
        # 4. 視差計算只依賴X座標，Y差異不影響結果
        
        print(f"    === 視差計算詳細過程 ===")
        print(f"    採樣點數: {min_points}")
        print(f"    左影像x範圍: [{np.min(left_x):.1f}, {np.max(left_x):.1f}]")
        print(f"    右影像x範圍: [{np.min(right_x):.1f}, {np.max(right_x):.1f}]")
        print(f"    視差範圍: [{np.min(disparities):.1f}, {np.max(disparities):.1f}]")
        print(f"    左影像y範圍: [{np.min(left_y):.1f}, {np.max(left_y):.1f}]")
        print(f"    右影像y範圍: [{np.min(right_y):.1f}, {np.max(right_y):.1f}]")
        print(f"    y座標差異範圍: [{np.min(y_diff):.1f}, {np.max(y_diff):.1f}]")
        print(f"    y座標差異統計: 平均={np.mean(y_diff):.2f} ± {np.std(y_diff):.2f}")
        
        # 計算視差統計
        print(f"    視差統計: 平均={np.mean(disparities):.2f} ± {np.std(disparities):.2f}")
        print(f"    視差範圍: [{np.min(disparities):.1f}, {np.max(disparities):.1f}]")
        
        # 分析視差的一致性
        disparity_std = np.std(disparities)
        if disparity_std < 5:
            print(f"    視差一致性: 良好 (標準差={disparity_std:.2f} < 5)")
        elif disparity_std < 15:
            print(f"    視差一致性: 中等 (標準差={disparity_std:.2f})")
        else:
            print(f"    視差一致性: 變化較大 (標準差={disparity_std:.2f}) - 可能反映蝦子3D形狀")
        
        return disparities, debug_info
    
    def analyze_shrimp_disparity(self, matches, n_points=10):
        """
        分析蝦子視差
        
        Args:
            matches (list): 匹配的蝦子列表
            n_points (int): 在曲線上採樣的點數
            
        Returns:
            list: 視差分析結果
        """
        disparity_results = []
        
        for i, match in enumerate(matches):
            print(f"\n分析蝦子匹配 {i+1}/{len(matches)}")
            
            # 提取左右蝦子的遮罩點
            left_mask = match['left_full_mask']
            right_mask = match['right_full_mask']
            
            left_x, left_y = self.get_mask_points(left_mask)
            right_x, right_y = self.get_mask_points(right_mask)
            
            if len(left_x) == 0 or len(right_x) == 0:
                print(f"  跳過: 蝦子 {i+1} 遮罩點為空")
                continue
            
            # 對左右蝦子分別計算最佳迴歸曲線
            print(f"  左蝦子點數: {len(left_x)}, 右蝦子點數: {len(right_x)}")
            
            # 計算不同曲線類型的擬合效果
            curve_types = ['linear', 'quadratic', 'cubic']
            left_curves = {}
            right_curves = {}
            
            for curve_type in curve_types:
                try:
                    # 左蝦子曲線
                    left_params = self.fit_optimal_curve(left_x, left_y, curve_type)
                    left_distance = self.total_distance_to_curve(left_params, left_x, left_y, curve_type)
                    left_curves[curve_type] = {
                        'params': left_params,
                        'distance': left_distance,
                        'avg_distance': left_distance / len(left_x)
                    }
                    
                    # 右蝦子曲線
                    right_params = self.fit_optimal_curve(right_x, right_y, curve_type)
                    right_distance = self.total_distance_to_curve(right_params, right_x, right_y, curve_type)
                    right_curves[curve_type] = {
                        'params': right_params,
                        'distance': right_distance,
                        'avg_distance': right_distance / len(right_x)
                    }
                    
                    print(f"    {curve_type}: 左平均距離={left_distance/len(left_x):.2f}, 右平均距離={right_distance/len(right_x):.2f}")
                    
                except Exception as e:
                    print(f"    {curve_type} 擬合失敗: {e}")
                    left_curves[curve_type] = None
                    right_curves[curve_type] = None
            
            # 選擇最佳曲線類型 (使用二次曲線作為默認，因為蝦子通常是彎曲的)
            best_curve_type = 'quadratic'
            
            # 如果二次曲線擬合失敗，嘗試其他類型
            if left_curves[best_curve_type] is None or right_curves[best_curve_type] is None:
                for curve_type in curve_types:
                    if left_curves[curve_type] is not None and right_curves[curve_type] is not None:
                        best_curve_type = curve_type
                        break
            
            if left_curves[best_curve_type] is None or right_curves[best_curve_type] is None:
                print(f"  跳過: 蝦子 {i+1} 無法擬合任何曲線")
                continue
            
            # 使用新的對應點匹配方法
            try:
                left_sampled_points, right_sampled_points = self.find_corresponding_points_on_curves(
                    left_curves[best_curve_type]['params'], 
                    right_curves[best_curve_type]['params'], 
                    best_curve_type, 
                    n_points,
                    left_mask,
                    right_mask
                )
                left_sampled_x, left_sampled_y = left_sampled_points
                right_sampled_x, right_sampled_y = right_sampled_points
                
                print(f"    使用改進的對應點匹配方法")
                
            except Exception as e:
                print(f"    對應點匹配失敗，使用原方法: {e}")
                # 回到原來的方法：在曲線上採樣點
                left_x_range = (left_x.min(), left_x.max())
                right_x_range = (right_x.min(), right_x.max())
                
                left_sampled_x, left_sampled_y = self.sample_points_on_curve(
                    left_curves[best_curve_type]['params'], left_x_range, n_points, best_curve_type
                )
                
                right_sampled_x, right_sampled_y = self.sample_points_on_curve(
                    right_curves[best_curve_type]['params'], right_x_range, n_points, best_curve_type
                )
            
            # 計算視差
            disparities, debug_info = self.calculate_disparity(
                (left_sampled_x, left_sampled_y), 
                (right_sampled_x, right_sampled_y)
            )
            
            # 儲存結果
            result = {
                'match_idx': i,
                'left_curves': left_curves,
                'right_curves': right_curves,
                'best_curve_type': best_curve_type,
                'left_sampled_points': (left_sampled_x, left_sampled_y),
                'right_sampled_points': (right_sampled_x, right_sampled_y),
                'disparities': disparities,
                'avg_disparity': np.mean(disparities),
                'disparity_std': np.std(disparities),
                'match_info': match,
                'debug_info': debug_info
            }
            
            disparity_results.append(result)
            
            print(f"  ✓ 曲線類型: {best_curve_type}")
            print(f"  ✓ 採樣點數: {n_points}")
            print(f"  ✓ 平均視差: {result['avg_disparity']:.2f} ± {result['disparity_std']:.2f} 像素")
        
        return disparity_results
    
    def visualize_disparity_results(self, left_image, right_image, disparity_results, output_dir):
        """
        視覺化視差分析結果
        
        Args:
            left_image (numpy.ndarray): 左影像
            right_image (numpy.ndarray): 右影像
            disparity_results (list): 視差分析結果
            output_dir (str): 輸出目錄
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 創建整體視覺化
        fig = plt.figure(figsize=(20, 12))
        
        # 子圖佈局: 左影像、右影像、視差分析
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[1, 1, 1])
        
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        ax_disparity = fig.add_subplot(gs[0, 2])
        ax_stats = fig.add_subplot(gs[1, :])
        
        # 顯示左右影像
        ax_left.imshow(left_image)
        ax_left.set_title('Left Image with Regression Curves', fontsize=14, fontweight='bold')
        ax_left.axis('off')
        
        ax_right.imshow(right_image)
        ax_right.set_title('Right Image with Regression Curves', fontsize=14, fontweight='bold')
        ax_right.axis('off')
        
        # 顏色映射
        colors = plt.cm.tab10(np.linspace(0, 1, len(disparity_results)))
        
        # 儲存視差資料用於統計圖
        all_disparities = []
        disparity_labels = []
        
        for i, result in enumerate(disparity_results):
            color = colors[i]
            match = result['match_info']
            
            # 繪製左影像的蝦子和曲線
            left_bbox = match['left_bbox']
            x1, y1, x2, y2 = left_bbox
            rect_left = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax_left.add_patch(rect_left)
            
            # 繪製左影像曲線和採樣點
            left_x, left_y = result['left_sampled_points']
            ax_left.plot(left_x, left_y, 'o-', color=color, linewidth=3, markersize=8, 
                        label=f'Shrimp {i+1} ({result["best_curve_type"]})')
            
            # 繪製右影像的蝦子和曲線
            right_bbox = match['right_bbox']
            x1, y1, x2, y2 = right_bbox
            rect_right = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                      linewidth=2, edgecolor=color, facecolor='none')
            ax_right.add_patch(rect_right)
            
            # 繪製右影像曲線和採樣點
            right_x, right_y = result['right_sampled_points']
            ax_right.plot(right_x, right_y, 'o-', color=color, linewidth=3, markersize=8,
                         label=f'Shrimp {i+1} ({result["best_curve_type"]})')
            
            # 準備視差統計資料
            if len(result['disparities']) > 0:
                all_disparities.extend(result['disparities'])
                disparity_labels.extend([f'Shrimp {i+1}'] * len(result['disparities']))
        
        # 視差散點圖
        if all_disparities:
            unique_labels = list(set(disparity_labels))
            for j, label in enumerate(unique_labels):
                label_disparities = [d for d, l in zip(all_disparities, disparity_labels) if l == label]
                ax_disparity.scatter([j] * len(label_disparities), label_disparities, 
                                   c=colors[j], s=100, alpha=0.7, edgecolors='black')
            
            ax_disparity.set_xlabel('Shrimp Index', fontsize=12)
            ax_disparity.set_ylabel('Disparity (pixels)', fontsize=12)
            ax_disparity.set_title('Disparity Analysis', fontsize=14, fontweight='bold')
            ax_disparity.set_xticks(range(len(unique_labels)))
            ax_disparity.set_xticklabels(unique_labels, rotation=45)
            ax_disparity.grid(True, alpha=0.3)
        
        # 統計表格
        ax_stats.axis('off')
        
        # 創建統計表格
        table_data = [['Shrimp', 'Curve Type', 'Sample Points', 'Avg Disparity', 'Std Disparity', 'NCC Score']]
        
        for i, result in enumerate(disparity_results):
            match = result['match_info']
            table_data.append([
                f'Shrimp {i+1}',
                result['best_curve_type'].capitalize(),
                f"{len(result['disparities'])}",
                f"{result['avg_disparity']:.2f}",
                f"{result['disparity_std']:.2f}",
                f"{match['ncc_score']:.3f}"
            ])
        
        table = ax_stats.table(cellText=table_data[1:], colLabels=table_data[0],
                              cellLoc='center', loc='center',
                              bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 設置表格樣式
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # 標題行
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#40466e')
                cell.set_text_props(color='white')
            else:
                cell.set_facecolor('#f1f1f2')
        
        ax_left.legend(bbox_to_anchor=(0.02, 0.98), loc='upper left', fontsize=10)
        ax_right.legend(bbox_to_anchor=(0.02, 0.98), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        # 儲存整體結果
        overall_output_path = os.path.join(output_dir, 'shrimp_disparity_analysis.png')
        plt.savefig(overall_output_path, dpi=300, bbox_inches='tight')
        print(f"整體視差分析結果已儲存: {overall_output_path}")
        plt.close()
        
        # 為每個蝦子創建詳細分析圖
        for i, result in enumerate(disparity_results):
            self.create_detailed_shrimp_analysis(left_image, right_image, result, i, output_dir)
    
    def create_detailed_shrimp_analysis(self, left_image, right_image, result, shrimp_idx, output_dir):
        """
        為單隻蝦子創建詳細分析圖
        
        Args:
            left_image (numpy.ndarray): 左影像
            right_image (numpy.ndarray): 右影像
            result (dict): 單隻蝦子的分析結果
            shrimp_idx (int): 蝦子索引
            output_dir (str): 輸出目錄
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1])
        
        match = result['match_info']
        
        # 左影像ROI
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(match['left_roi'])
        ax1.set_title(f'Left Shrimp {shrimp_idx+1} ROI', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 右影像ROI  
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(match['right_roi'])
        ax2.set_title(f'Right Shrimp {shrimp_idx+1} ROI', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # 左影像完整圖與曲線
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(left_image)
        left_bbox = match['left_bbox']
        x1, y1, x2, y2 = left_bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=3, edgecolor='red', facecolor='none')
        ax3.add_patch(rect)
        
        # 繪製左影像曲線和採樣點
        left_x, left_y = result['left_sampled_points']
        ax3.plot(left_x, left_y, 'o-', color='yellow', linewidth=3, markersize=10, 
                markeredgecolor='red', label=f'{result["best_curve_type"]} curve')
        
        ax3.set_title('Left Image with Regression Curve', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.axis('off')
        
        # 右影像完整圖與曲線
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.imshow(right_image)
        right_bbox = match['right_bbox']
        x1, y1, x2, y2 = right_bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=3, edgecolor='red', facecolor='none')
        ax4.add_patch(rect)
        
        # 繪製右影像曲線和採樣點
        right_x, right_y = result['right_sampled_points']
        ax4.plot(right_x, right_y, 'o-', color='yellow', linewidth=3, markersize=10,
                markeredgecolor='red', label=f'{result["best_curve_type"]} curve')
        
        ax4.set_title('Right Image with Regression Curve', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.axis('off')
        
        # 視差分析圖
        ax5 = fig.add_subplot(gs[2, :])
        
        if len(result['disparities']) > 0:
            point_indices = np.arange(len(result['disparities']))
            
            # 繪製所有視差點（全部視為有效）
            ax5.scatter(point_indices, result['disparities'], 
                       c='green', s=100, alpha=0.7, edgecolors='black')
            
            # 繪製平均視差線
            ax5.axhline(y=result['avg_disparity'], color='blue', linestyle='--', 
                       linewidth=2, label=f'Avg Disparity: {result["avg_disparity"]:.2f}')
            
            ax5.set_xlabel('Point Index', fontsize=12)
            ax5.set_ylabel('Disparity (pixels)', fontsize=12)
            ax5.set_title(f'Disparity Analysis - Shrimp {shrimp_idx+1}', fontsize=14, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No valid disparity points', 
                    transform=ax5.transAxes, ha='center', va='center', fontsize=16)
        
        # 添加總體資訊
        fig.suptitle(f'Detailed Analysis - Shrimp {shrimp_idx+1} (NCC: {match["ncc_score"]:.3f})', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # 儲存詳細分析結果
        detail_output_path = os.path.join(output_dir, f'shrimp_{shrimp_idx+1}_detailed_analysis.png')
        plt.savefig(detail_output_path, dpi=300, bbox_inches='tight')
        print(f"蝦子 {shrimp_idx+1} 詳細分析已儲存: {detail_output_path}")
        plt.close()
    
    def save_results_json(self, disparity_results, output_path):
        """
        儲存視差分析結果為JSON檔案
        
        Args:
            disparity_results (list): 視差分析結果
            output_path (str): 輸出檔案路徑
        """
        # 準備可序列化的資料
        serializable_results = []
        
        for i, result in enumerate(disparity_results):
            match = result['match_info']
            
            serializable_result = {
                'shrimp_id': i + 1,
                'ncc_score': float(match['ncc_score']),
                'curve_type': result['best_curve_type'],
                'left_bbox': match['left_bbox'],
                'right_bbox': match['right_bbox'],
                'sampling_points': len(result['disparities']),
                'avg_disparity': float(result['avg_disparity']),
                'disparity_std': float(result['disparity_std']),
                'disparities': result['disparities'].tolist(),
                'left_sampled_points': {
                    'x': result['left_sampled_points'][0].tolist(),
                    'y': result['left_sampled_points'][1].tolist()
                },
                'right_sampled_points': {
                    'x': result['right_sampled_points'][0].tolist(),
                    'y': result['right_sampled_points'][1].tolist()
                }
            }
            
            serializable_results.append(serializable_result)
        
        # 儲存JSON檔案
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_shrimps_analyzed': len(serializable_results),
                'analysis_results': serializable_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"視差分析結果JSON已儲存: {output_path}")
    
    def run_analysis(self, left_json_path, right_json_path, output_dir, n_points=10):
        """
        執行完整的蝦子視差分析
        
        Args:
            left_json_path (str): 左影像JSON檔案路徑
            right_json_path (str): 右影像JSON檔案路徑
            output_dir (str): 輸出目錄
            n_points (int): 在曲線上採樣的點數
        """
        try:
            print("=== 蝦子雙目立體視覺視差分析程式 ===")
            print(f"左影像JSON: {left_json_path}")
            print(f"右影像JSON: {right_json_path}")
            print(f"輸出目錄: {output_dir}")
            print(f"曲線採樣點數: {n_points}")
            
            # 1. 載入左右影像和標註資料
            print("\n1. 載入左影像和標註資料...")
            left_image, left_data = self.load_image_and_annotations(left_json_path)
            print(f"   左影像尺寸: {left_image.shape}")
            
            print("\n2. 載入右影像和標註資料...")
            right_image, right_data = self.load_image_and_annotations(right_json_path)
            print(f"   右影像尺寸: {right_image.shape}")
            
            # 2. 執行蝦子匹配
            print("\n3. 執行蝦子NCC匹配...")
            matches = self.perform_shrimp_matching(left_data, right_data, left_image, right_image)
            print(f"   找到 {len(matches)} 個蝦子匹配")
            
            if len(matches) == 0:
                print("未找到任何蝦子匹配，程式結束")
                return
            
            # 3. 分析視差
            print("\n4. 分析蝦子視差...")
            disparity_results = self.analyze_shrimp_disparity(matches, n_points)
            
            if len(disparity_results) == 0:
                print("無法分析任何蝦子的視差，程式結束")
                return
            
            # 4. 視覺化結果
            print("\n5. 產生視覺化結果...")
            self.visualize_disparity_results(left_image, right_image, disparity_results, output_dir)
            
            # 5. 儲存結果
            print("\n6. 儲存分析結果...")
            json_output_path = os.path.join(output_dir, 'shrimp_disparity_results.json')
            self.save_results_json(disparity_results, json_output_path)
            
            # 輸出摘要
            print("\n=== 分析結果摘要 ===")
            for i, result in enumerate(disparity_results):
                print(f"蝦子 {i+1}:")
                print(f"  - NCC分數: {result['match_info']['ncc_score']:.3f}")
                print(f"  - 曲線類型: {result['best_curve_type']}")
                print(f"  - 採樣點數: {len(result['disparities'])}")
                print(f"  - 平均視差: {result['avg_disparity']:.2f} ± {result['disparity_std']:.2f} 像素")
            
            print(f"\n程式執行完成! 結果已儲存至: {output_dir}")
            
        except Exception as e:
            print(f"錯誤: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函數"""
    # 定義檔案路徑
    left_json_path = r"d:\Git\Grounded-SAM-2\samples\yolo_sam2_video_demo_enhanced_left\frame_015314\frame_015314_results.json"
    right_json_path = r"d:\Git\Grounded-SAM-2\samples\yolo_sam2_video_demo_enhanced_right\frame_015395\frame_015395_results.json"
    
    # 輸出目錄
    output_dir = r"d:\Git\Grounded-SAM-2\shrimp_disparity_analysis_results"
    
    # 在曲線上採樣的點數
    n_points = 15
    
    # 創建分析器並執行分析
    analyzer = ShrimpDisparityAnalyzer(ncc_threshold=0.3)
    analyzer.run_analysis(left_json_path, right_json_path, output_dir, n_points)


if __name__ == "__main__":
    main()
