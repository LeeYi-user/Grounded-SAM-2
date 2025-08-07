import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device, time_synchronized
import glob
from tqdm import tqdm

def underwater_enhancement(img):
    """
    水下影像還原處理
    參考 color_correction.py 的作法
    """
    # Step 1: 白平衡（簡單灰世界算法）
    def white_balance(img):
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result

    wb_image = white_balance(img)

    # Step 2: CLAHE 增強對比
    lab = cv2.cvtColor(wb_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    contrast_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Step 3: 銳化
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    sharpened = cv2.filter2D(contrast_image, -1, kernel_sharpening)
    
    return sharpened

"""
Hyper parameters
"""
IMG_DIR = "extracted_frames/"  # 處理整個資料夾
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
YOLO_CHECKPOINT = "checkpoints/shrimp_674.pt"  # 使用 YOLOv7 模型
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.45
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("outputs/yolo_sam2_video_demo")
DUMP_JSON_RESULTS = True

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# initialize logging and device selection
set_logging()
device = select_device(DEVICE)
half = device.type != 'cpu'  # 只有 CUDA 支援半精度

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# load YOLOv7 model
yolo_model = attempt_load(YOLO_CHECKPOINT, map_location=device)
stride = int(yolo_model.stride.max())
img_size = check_img_size(IMG_SIZE, s=stride)

if half:
    yolo_model.half()

# get class names
names = yolo_model.module.names if hasattr(yolo_model, 'module') else yolo_model.names
print(f"YOLOv7 detection classes: {names}")

# warmup YOLOv7 model
if device.type != 'cpu':
    yolo_model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(yolo_model.parameters())))

# get all image paths
image_paths = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
print(f"Found {len(image_paths)} images to process")

# process each image
for img_path in tqdm(image_paths, desc="Processing images"):
    print(f"\nProcessing: {img_path}")
    
    # create output directory for this image
    img_name = Path(img_path).stem
    img_output_dir = OUTPUT_DIR / img_name
    img_output_dir.mkdir(parents=True, exist_ok=True)
    
    # load image using OpenCV for SAM2
    image_source = cv2.imread(img_path)
    if image_source is None:
        print(f"Failed to load image: {img_path}")
        continue
    image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)

    # setup data loader for YOLOv7
    dataset = LoadImages(img_path, img_size=img_size, stride=stride)

    sam2_predictor.set_image(image_source)

    # run YOLOv7 inference
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0  # normalize to 0.0-1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # inference
        with torch.no_grad():
            pred = yolo_model(img, augment=False)[0]
        
        # apply NMS
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=None, agnostic=False)
        
        # process detections
        for i, det in enumerate(pred):
            if len(det):
                # rescale boxes from img_size to original image size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_source.shape).round()
                
                # extract boxes, confidences, and class names
                boxes_xyxy = det[:, :4].cpu().numpy()
                confidences = det[:, 4].cpu().numpy()
                class_ids = det[:, 5].cpu().numpy().astype(int)
                class_names = [names[int(cls)] for cls in class_ids]
                
                # convert to format expected by SAM2 (xyxy format is already correct)
                input_boxes = boxes_xyxy
            else:
                print(f"No objects detected in {img_name}")
                input_boxes = np.array([]).reshape(0, 4)
                confidences = np.array([])
                class_names = []
                class_ids = np.array([])

    # Only proceed with SAM2 if objects were detected
    if len(input_boxes) > 0:
        print(f"Detected {len(input_boxes)} objects, applying underwater enhancement...")
        
        # Apply underwater enhancement to the entire image
        enhanced_image = underwater_enhancement(cv2.imread(img_path))
        enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
        
        # Get SAM2 results on enhanced image
        print("Getting SAM2 results on enhanced image...")
        sam2_predictor.set_image(enhanced_image_rgb)  # Use enhanced image
        
        torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        enhanced_masks, enhanced_scores, enhanced_logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if enhanced_masks.ndim == 4:
            enhanced_masks = enhanced_masks.squeeze(1)

        # get image dimensions
        h, w = image_source.shape[:2]

        confidences = confidences.tolist()

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]

        """
        Visualize image with supervision useful API
        """
        # Save original image with enhanced masks (main result)
        original_img = cv2.imread(img_path)
        enhanced_mask_on_original_detections = sv.Detections(
            xyxy=input_boxes,
            mask=enhanced_masks.astype(bool),  # Using enhanced masks on original image
            class_id=class_ids
        )
        
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()
        
        original_with_enhanced_mask = box_annotator.annotate(scene=original_img.copy(), detections=enhanced_mask_on_original_detections)
        original_with_enhanced_mask = label_annotator.annotate(scene=original_with_enhanced_mask, detections=enhanced_mask_on_original_detections, labels=labels)
        original_with_enhanced_mask = mask_annotator.annotate(scene=original_with_enhanced_mask, detections=enhanced_mask_on_original_detections)
        cv2.imwrite(os.path.join(img_output_dir, f"{img_name}_result.jpg"), original_with_enhanced_mask)

        """
        Dump the results in standard format and save as json files
        """

        def single_mask_to_rle(mask):
            rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        if DUMP_JSON_RESULTS:
            # convert mask into rle format
            enhanced_mask_rles = [single_mask_to_rle(mask) for mask in enhanced_masks]

            input_boxes = input_boxes.tolist()
            enhanced_scores = enhanced_scores.tolist()
            
            # save the results in standard format
            results = {
                "image_path": img_path,
                "annotations" : [
                    {
                        "class_name": class_name,
                        "bbox": box,
                        "segmentation": mask_rle,
                        "score": score,
                    }
                    for class_name, box, mask_rle, score in zip(class_names, input_boxes, enhanced_mask_rles, enhanced_scores)
                ],
                "box_format": "xyxy",
                "img_width": w,
                "img_height": h,
            }

            with open(os.path.join(img_output_dir, f"{img_name}_results.json"), "w") as f:
                json.dump(results, f, indent=4)

    else:
        print(f"No objects detected in {img_name}, skipping SAM2 segmentation")

print(f"\nBatch processing completed! Results saved in: {OUTPUT_DIR}")
