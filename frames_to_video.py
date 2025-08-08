#!/usr/bin/env python3
"""
簡化版本：將 outputs/yolo_sam2_video_demo 下的frame圖片合併成MP4影片
固定設定：29.97 FPS，輸出到當前目錄的 yolo_sam2_video_output.mp4
"""

import os
import cv2
import glob
import re
from tqdm import tqdm


def main():
    # 固定設定
    input_dir = r"outputs\yolo_sam2_video_demo_enhanced_left"
    output_file = "yolo_sam2_video_output_enhanced_left.mp4"
    fps = 29.97
    
    print("=== YOLO SAM2 Frame 轉 MP4 影片工具 ===")
    print(f"輸入目錄: {input_dir}")
    print(f"輸出檔案: {output_file}")
    print(f"影片幀率: {fps} FPS")
    print("-" * 50)
    
    # 檢查輸入目錄
    if not os.path.exists(input_dir):
        print(f"❌ 錯誤: 輸入目錄不存在: {input_dir}")
        return
    
    # 獲取所有frame資料夾
    print("🔍 掃描frame資料夾...")
    frame_folders = []
    for item in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, item)) and item.startswith('frame_'):
            frame_folders.append(item)
    
    if not frame_folders:
        print("❌ 沒有找到任何frame資料夾!")
        return
    
    # 按frame編號排序
    def get_frame_number(folder_name):
        match = re.search(r'frame_(\d+)', folder_name)
        return int(match.group(1)) if match else 0
    
    frame_folders.sort(key=get_frame_number)
    print(f"📁 找到 {len(frame_folders)} 個frame資料夾")
    
    # 收集所有圖片檔案
    print("🖼️  收集frame圖片...")
    frame_files = []
    for folder in frame_folders:
        folder_path = os.path.join(input_dir, folder)
        # 尋找該資料夾下的result.jpg檔案
        result_files = glob.glob(os.path.join(folder_path, "*_result.jpg"))
        if result_files:
            frame_files.append(result_files[0])
    
    if not frame_files:
        print("❌ 沒有找到任何result.jpg圖片檔案!")
        return
    
    print(f"🎯 找到 {len(frame_files)} 個有效圖片檔案")
    
    # 讀取第一張圖片獲取尺寸
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"❌ 無法讀取第一張圖片: {frame_files[0]}")
        return
    
    height, width = first_frame.shape[:2]
    print(f"📏 影片尺寸: {width} x {height}")
    
    # 建立影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("❌ 無法建立影片寫入器!")
        return
    
    # 寫入frame到影片
    print("🎬 正在合併frame...")
    success_count = 0
    
    for i, frame_file in enumerate(tqdm(frame_files, desc="合併進度")):
        frame = cv2.imread(frame_file)
        if frame is not None:
            # 確保尺寸一致
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            video_writer.write(frame)
            success_count += 1
        else:
            print(f"⚠️  警告: 無法讀取 {frame_file}")
    
    # 清理資源
    video_writer.release()
    cv2.destroyAllWindows()
    
    # 完成報告
    duration = success_count / fps
    print("-" * 50)
    print("✅ 影片合併完成!")
    print(f"📊 統計資訊:")
    print(f"   - 處理的frame數: {success_count}")
    print(f"   - 影片時長: {duration:.2f} 秒")
    print(f"   - 影片檔案: {output_file}")
    
    # 檢查檔案大小
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        print(f"   - 檔案大小: {file_size:.1f} MB")
    
    print(f"🎉 影片已儲存至: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()
