import os
import cv2
import json
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from flask import current_app

class_colors = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 255, 255)}

def generate_palette(n):
    cmap = plt.get_cmap('tab20')
    return (np.array([cmap(i % 20)[:3] for i in range(n)]) * 255).astype(np.uint8)

def assign_tooth_numbers_by_bbox(tooth_entries):
    boxes = [(i, entry["bbox"]["x1"]) for i, entry in enumerate(tooth_entries)]
    boxes_sorted = sorted(boxes, key=lambda x: x[1])
    return {idx: i + 1 for i, (idx, _) in enumerate(boxes_sorted)}

def smooth_binary_mask(mask_bin, epsilon=2.0):
    mask_uint8 = (mask_bin * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(mask_uint8, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smoothed = np.zeros_like(mask_uint8)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=True)
        hull = cv2.convexHull(approx)
        cv2.drawContours(smoothed, [hull], -1, 255, -1)
    return smoothed > 127

def run_yolo_segmentation(image_path: str) -> dict:
    model = YOLO(current_app.config['YOLO_MODEL_PATH'])
    class_names = model.names

    mask_classID_dir = current_app.config['PROCESSED_FOLDER_MODEL3_1']
    mask_classID_tooth_dir = current_app.config['PROCESSED_FOLDER_MODEL3_2']

    os.makedirs(mask_classID_dir, exist_ok=True)
    os.makedirs(mask_classID_tooth_dir, exist_ok=True)

    base_name = Path(image_path).stem
    orig_img = cv2.imread(image_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img_h, img_w = orig_img.shape[:2]

    results = model.predict(source=image_path, save=False, imgsz=640, conf=0.25, verbose=False)
    r = results[0]

    mask_classID = orig_img.copy()
    mask_classID_tooth = orig_img.copy()

    if r.masks is not None and getattr(r.masks, 'data', None) is not None:
        mask_data = r.masks.data
        palette_index = generate_palette(len(mask_data))

        all_masks = []
        tooth_indices = []

        for i, mask in enumerate(mask_data):
            mask_np = mask.cpu().numpy()
            resized_mask = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            mask_bin = resized_mask > 0.5
            mask_bin = smooth_binary_mask(mask_bin)  # ✅ 후처리 적용

            cls_id = int(r.boxes.cls[i]) if r.boxes is not None and len(r.boxes.cls) > i else -1
            color = class_colors.get(cls_id, (128, 128, 128))

            class_mask = np.zeros_like(orig_img, dtype=np.uint8)
            for c in range(3):
                class_mask[:, :, c][mask_bin] = color[c]
            mask_classID = cv2.addWeighted(mask_classID, 1.0, class_mask, 0.5, 0)

            all_masks.append({'index': i, 'class_id': cls_id, 'mask': mask_bin})
            if cls_id == 3:
                tooth_indices.append(i)

        palette_tooth = generate_palette(len(tooth_indices))
        merged_masks = [all_masks[i]['mask'].copy() for i in tooth_indices]

        for obj in all_masks:
            if obj['class_id'] == 3:
                continue
            best_idx = -1
            best_overlap = 0
            for t_order, t_idx in enumerate(tooth_indices):
                overlap = np.sum(np.logical_and(all_masks[t_idx]['mask'], obj['mask']))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = t_order
            if best_idx >= 0:
                merged_masks[best_idx] = np.logical_or(merged_masks[best_idx], obj['mask'])

        json_list = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = class_names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            json_list.append({
                'class_id': cls_id,
                'class_name': cls_name,
                'confidence': conf,
                'bbox': {'x1': xyxy[0], 'y1': xyxy[1], 'x2': xyxy[2], 'y2': xyxy[3]}
            })

        tooth_entries = [e for e in json_list if e['class_id'] == 3]
        numbering_map = assign_tooth_numbers_by_bbox(tooth_entries)
        for i, entry in enumerate(tooth_entries):
            number = numbering_map.get(i, "?")
            entry["tooth_number"] = number

        for t_order, merged_mask in enumerate(merged_masks):
            color = palette_tooth[t_order]
            overlay = np.zeros_like(orig_img, dtype=np.uint8)
            for c in range(3):
                overlay[:, :, c][merged_mask] = color[c]
            mask_classID_tooth = cv2.addWeighted(mask_classID_tooth, 1.0, overlay, 0.5, 0)

        disease_reports = {}
        for lesion in json_list:
            if lesion['class_id'] == 3:
                continue
            cx = (lesion['bbox']['x1'] + lesion['bbox']['x2']) / 2
            cy = (lesion['bbox']['y1'] + lesion['bbox']['y2']) / 2
            best_tooth = None
            best_iou = 0
            for tooth in tooth_entries:
                tx1, ty1 = tooth['bbox']['x1'], tooth['bbox']['y1']
                tx2, ty2 = tooth['bbox']['x2'], tooth['bbox']['y2']
                if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                    best_tooth = tooth
                    break
                else:
                    ix1, iy1 = max(lesion['bbox']['x1'], tx1), max(lesion['bbox']['y1'], ty1)
                    ix2, iy2 = min(lesion['bbox']['x2'], tx2), min(lesion['bbox']['y2'], ty2)
                    iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
                    intersection = iw * ih
                    lesion_area = (lesion['bbox']['x2'] - lesion['bbox']['x1']) * (lesion['bbox']['y2'] - lesion['bbox']['y1'])
                    tooth_area = (tx2 - tx1) * (ty2 - ty1)
                    union = lesion_area + tooth_area - intersection
                    iou = intersection / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_tooth = tooth
            if best_tooth:
                t_num = best_tooth.get("tooth_number", "?")
                disease = lesion.get("class_name", f"class_{lesion['class_id']}")
                disease_reports.setdefault(t_num, []).append(disease)

        diagnosis_summary = [
            f"{t_num}번 치아에 {' / '.join(set(diseases))}가 있습니다."
            for t_num, diseases in sorted(disease_reports.items())
        ]

        model3_1_path = os.path.join(mask_classID_dir, f"{base_name}.png")
        model3_2_path = os.path.join(mask_classID_tooth_dir, f"{base_name}.png")
        Image.fromarray(mask_classID).save(model3_1_path)
        Image.fromarray(mask_classID_tooth).save(model3_2_path)

        return {
            "diagnosis_summary": diagnosis_summary,
            "predictions": json_list,
            "model3_1_path": model3_1_path,
            "model3_2_path": model3_2_path,
            "model3_1_message": "model3_1 마스크 생성 완료",
            "model3_2_message": "model3_2 마스크 생성 완료"
        }

    return {
        "diagnosis_summary": [],
        "predictions": [],
        "model3_1_path": "",
        "model3_2_path": "",
        "model3_1_message": "마스크 없음",
        "model3_2_message": "마스크 없음"
    }
