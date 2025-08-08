#!/usr/bin/env python3
"""
相對位置採樣示例程式
用簡單的例子說明如何在蝦子曲線上進行相對位置採樣
"""

import numpy as np
import matplotlib.pyplot as plt

# 解決中文顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def demonstrate_relative_sampling():
    """演示相對位置採樣的概念"""
    
    print("=== 相對位置採樣示例 ===\n")
    
    # 模擬左右蝦子的範圍
    left_x_min, left_x_max = 700, 900    # 左蝦子x範圍
    right_x_min, right_x_max = 650, 850  # 右蝦子x範圍（整體向左偏移50像素）
    
    print(f"左蝦子x範圍: [{left_x_min}, {left_x_max}] (長度: {left_x_max - left_x_min})")
    print(f"右蝦子x範圍: [{right_x_min}, {right_x_max}] (長度: {right_x_max - right_x_min})")
    
    # 模擬蝦子的曲線方程（簡化為二次曲線）
    # 左蝦子: y = -0.001 * (x-800)² + 400 (在x=800處達到最高點)
    # 右蝦子: y = -0.001 * (x-750)² + 410 (在x=750處達到最高點，稍微高一點)
    
    def left_curve(x):
        return -0.001 * (x - 800)**2 + 400
    
    def right_curve(x):
        return -0.001 * (x - 750)**2 + 410
    
    # 設定採樣點數
    n_points = 5
    
    print(f"\n採樣點數: {n_points}")
    print("=" * 50)
    
    # 方法1：錯誤的傳統方法（固定x座標採樣）
    print("\n【方法1：傳統方法 - 錯誤】")
    print("在左右影像中使用相同的x座標範圍採樣")
    
    # 使用共同的x範圍
    common_x_min = max(left_x_min, right_x_min)  # 650
    common_x_max = min(left_x_max, right_x_max)  # 850
    traditional_x = np.linspace(common_x_min, common_x_max, n_points)
    
    traditional_left_y = left_curve(traditional_x)
    traditional_right_y = right_curve(traditional_x)
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
    
    print(f"\n平均視差: {np.mean(relative_disparities):.1f} 像素 ← 正確！")
    print(f"視差標準差: {np.std(relative_disparities):.1f} 像素")
    
    # 視覺化結果
    create_visualization(left_x_min, left_x_max, right_x_min, right_x_max,
                        left_curve, right_curve, 
                        relative_left_x, relative_left_y,
                        relative_right_x, relative_right_y,
                        relative_disparities)

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
    
    plt.plot(x_range, left_y_range, 'b-', linewidth=2, label='左蝦子曲線', alpha=0.3)
    plt.plot(x_range, right_y_range, 'r-', linewidth=2, label='右蝦子曲線', alpha=0.3)
    
    # 標出蝦子的有效範圍
    left_valid_x = np.linspace(left_x_min, left_x_max, 50)
    right_valid_x = np.linspace(right_x_min, right_x_max, 50)
    plt.plot(left_valid_x, left_curve(left_valid_x), 'b-', linewidth=4, label='左蝦子')
    plt.plot(right_valid_x, right_curve(right_valid_x), 'r-', linewidth=4, label='右蝦子')
    
    # 標出採樣點
    plt.scatter(sample_left_x, sample_left_y, c='blue', s=100, zorder=5, marker='o')
    plt.scatter(sample_right_x, sample_right_y, c='red', s=100, zorder=5, marker='s')
    
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
    plt.plot(point_indices, sample_left_x, 'bo-', label='左影像 x座標', linewidth=2, markersize=8)
    plt.plot(point_indices, sample_right_x, 'ro-', label='右影像 x座標', linewidth=2, markersize=8)
    plt.xlabel('採樣點編號')
    plt.ylabel('X 座標 (像素)')
    plt.title('左右影像 X 座標比較')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子圖4：y座標比較
    plt.subplot(2, 2, 4)
    plt.plot(point_indices, sample_left_y, 'bo-', label='左影像 y座標', linewidth=2, markersize=8)
    plt.plot(point_indices, sample_right_y, 'ro-', label='右影像 y座標', linewidth=2, markersize=8)
    plt.xlabel('採樣點編號')
    plt.ylabel('Y 座標 (像素)')
    plt.title('左右影像 Y 座標比較')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('相對位置採樣示例.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n視覺化圖表已儲存為: 相對位置採樣示例.png")

if __name__ == "__main__":
    demonstrate_relative_sampling()
