import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


def predict_shrimp_image():
    """
    使用 shrimp_674.pt 模型預測 customdata_830\images\vaild\00031.jpg 圖片
    """
    # 設定參數
    weights = 'checkpoints/shrimp_674.pt'
    source = r'extracted_frames_left\frame_015314.jpg'
    img_size = 640
    conf_thres = 0.25
    iou_thres = 0.45
    device = ''  # 自動選擇 GPU 或 CPU
    save_img = True
    view_img = False
    
    print(f"載入模型: {weights}")
    print(f"預測圖片: {source}")
    print("-" * 50)
    
    # 創建輸出資料夾
    save_dir = Path('runs/predict_shrimp')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # 只有 CUDA 支援半精度
    
    # 載入模型
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    
    if half:
        model.half()
    
    # 設定資料載入器
    dataset = LoadImages(source, img_size=img_size, stride=stride)
    
    # 取得類別名稱和顏色
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    print(f"偵測類別: {names}")
    print("-" * 50)
    
    # 預熱模型
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))
    
    t0 = time.time()
    
    # 執行推論
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0  # 正規化到 0.0-1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # 推論
        t1 = time_synchronized()
        with torch.no_grad():
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()
        
        # 應用 NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
        t3 = time_synchronized()
        
        # 處理偵測結果
        for i, det in enumerate(pred):
            p, s, im0 = path, '', im0s
            
            p = Path(p)
            save_path = str(save_dir / p.name)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 正規化增益 whwh
            
            if len(det):
                # 將邊界框從 img_size 縮放到 im0 大小
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # 印出結果
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                
                print(f"偵測到: {s}")
                
                # 畫出結果
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                    
                    # 印出詳細資訊
                    x1, y1, x2, y2 = xyxy
                    print(f"  - {names[int(cls)]}: 信心度 {conf:.3f}, 座標 ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
            
            else:
                print("未偵測到任何物件")
            
            # 印出時間
            print(f'推論時間: {(1E3 * (t2 - t1)):.1f}ms, NMS時間: {(1E3 * (t3 - t2)):.1f}ms')
            
            # 顯示結果
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)
            
            # 儲存結果
            if save_img:
                cv2.imwrite(save_path, im0)
                print(f"結果已儲存至: {save_path}")
    
    print(f'\n總執行時間: {time.time() - t0:.3f}s')
    print(f"預測完成！結果保存在 {save_dir} 資料夾中")


if __name__ == '__main__':
    predict_shrimp_image()
