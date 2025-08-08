#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量替換 JSON 檔案中的 image_path 路徑
將 "extracted_frames\\" 替換為 "extracted_frames_right\\"

用法：
python batch_replace_image_paths.py

作者: GitHub Copilot
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_replace_paths.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def find_json_files(directory: str) -> List[Path]:
    """
    遞迴尋找指定目錄下的所有 JSON 檔案
    
    Args:
        directory: 要搜尋的目錄路徑
        
    Returns:
        JSON 檔案路徑列表
    """
    json_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logger.warning(f"目錄不存在: {directory}")
        return json_files
    
    # 遞迴尋找所有 .json 檔案
    for json_file in directory_path.rglob("*.json"):
        json_files.append(json_file)
    
    logger.info(f"在 {directory} 中找到 {len(json_files)} 個 JSON 檔案")
    return json_files


def replace_image_path_in_json(json_file_path: Path, old_path: str, new_path: str) -> bool:
    """
    在 JSON 檔案中替換 image_path 欄位的路徑
    
    Args:
        json_file_path: JSON 檔案路徑
        old_path: 要被替換的舊路徑
        new_path: 新的路徑
        
    Returns:
        是否成功修改
    """
    try:
        # 讀取 JSON 檔案
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 檢查是否有修改
        modified = False
        
        # 處理不同的 JSON 結構
        if isinstance(data, dict):
            modified = replace_in_dict(data, old_path, new_path)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if replace_in_dict(item, old_path, new_path):
                        modified = True
        
        # 如果有修改，寫回檔案
        if modified:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"已更新: {json_file_path}")
            return True
        else:
            logger.debug(f"無需更新: {json_file_path}")
            return False
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析錯誤 {json_file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"處理檔案錯誤 {json_file_path}: {e}")
        return False


def replace_in_dict(data: Dict[str, Any], old_path: str, new_path: str) -> bool:
    """
    在字典中遞迴替換 image_path 欄位
    
    Args:
        data: 要處理的字典
        old_path: 要被替換的舊路徑
        new_path: 新的路徑
        
    Returns:
        是否有修改
    """
    modified = False
    
    for key, value in data.items():
        if key == "image_path" and isinstance(value, str):
            if old_path in value:
                data[key] = value.replace(old_path, new_path)
                modified = True
                logger.debug(f"替換路徑: {value} -> {data[key]}")
        elif isinstance(value, dict):
            if replace_in_dict(value, old_path, new_path):
                modified = True
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    if replace_in_dict(item, old_path, new_path):
                        modified = True
    
    return modified


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='批量替換 JSON 檔案中的 image_path 路徑')
    parser.add_argument('--old-path', default='extracted_frames\\', 
                       help='要被替換的舊路徑 (預設: extracted_frames\\)')
    parser.add_argument('--new-path', default='extracted_frames_right\\', 
                       help='新的路徑 (預設: extracted_frames_right\\)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='只顯示會被修改的檔案，不實際修改')
    args = parser.parse_args()
    
    # 目標目錄列表
    target_directories = [
        r"outputs\yolo_sam2_video_demo_right",
        r"outputs\yolo_sam2_video_demo_enhanced_right"
    ]
    
    old_path = args.old_path
    new_path = args.new_path
    
    logger.info(f"開始批量替換作業")
    logger.info(f"舊路徑: {old_path}")
    logger.info(f"新路徑: {new_path}")
    logger.info(f"乾燥運行模式: {args.dry_run}")
    
    total_files = 0
    total_modified = 0
    
    for target_dir in target_directories:
        if not os.path.exists(target_dir):
            logger.warning(f"目錄不存在，跳過: {target_dir}")
            continue
            
        logger.info(f"處理目錄: {target_dir}")
        
        # 尋找所有 JSON 檔案
        json_files = find_json_files(target_dir)
        total_files += len(json_files)
        
        # 處理每個 JSON 檔案
        modified_count = 0
        for json_file in json_files:
            if args.dry_run:
                # 乾燥運行模式：只檢查不修改
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if old_path in content:
                        logger.info(f"[乾燥運行] 會修改: {json_file}")
                        modified_count += 1
                except Exception as e:
                    logger.error(f"[乾燥運行] 讀取檔案錯誤 {json_file}: {e}")
            else:
                # 實際修改
                if replace_image_path_in_json(json_file, old_path, new_path):
                    modified_count += 1
        
        total_modified += modified_count
        logger.info(f"目錄 {target_dir} 處理完成: {modified_count}/{len(json_files)} 個檔案被修改")
    
    # 統計結果
    logger.info("=" * 50)
    logger.info("批量替換作業完成")
    logger.info(f"總共處理: {total_files} 個 JSON 檔案")
    logger.info(f"總共修改: {total_modified} 個檔案")
    
    if args.dry_run:
        logger.info("這是乾燥運行，未實際修改任何檔案")
        logger.info("如要實際執行，請移除 --dry-run 參數")


if __name__ == "__main__":
    main()
