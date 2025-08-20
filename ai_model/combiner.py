import numpy as np
from PIL import Image
from typing import List, Dict, Tuple

def get_overlap_ratios(mask_array: np.ndarray, bbox: List[float]) -> Tuple[float, float]:
    """
    마스크 픽셀이 바운딩 박스에 포함되는 비율과
    바운딩 박스 픽셀이 마스크에 포함되는 비율을 모두 계산합니다.
    """
    if mask_array.ndim != 2:
        return 0.0, 0.0

    x1, y1, x2, y2 = [int(round(c)) for c in bbox]

    # 바운딩 박스 영역을 정의합니다.
    y_min_bbox, y_max_bbox = max(0, y1), min(mask_array.shape[0], y2)
    x_min_bbox, x_max_bbox = max(0, x1), min(mask_array.shape[1], x2)
    
    if y_max_bbox <= y_min_bbox or x_max_bbox <= x_min_bbox:
        return 0.0, 0.0

    # 바운딩 박스 영역 내 마스크 픽셀 수 계산
    bbox_roi = mask_array[y_min_bbox:y_max_bbox, x_min_bbox:x_max_bbox]
    overlap_pixels = np.count_nonzero(bbox_roi)

    total_mask_pixels = np.count_nonzero(mask_array)
    bbox_area = (x_max_bbox - x_min_bbox) * (y_max_bbox - y_min_bbox)

    # 마스크 대 바운딩 박스 겹침 비율
    ratio_mask_in_bbox = overlap_pixels / total_mask_pixels if total_mask_pixels > 0 else 0.0
    
    # 바운딩 박스 대 마스크 겹침 비율
    ratio_bbox_in_mask = overlap_pixels / bbox_area if bbox_area > 0 else 0.0
    
    return ratio_mask_in_bbox, ratio_bbox_in_mask

def combine_results(
    original_image_size: Tuple[int, int],
    disease_results: List[Dict],
    hygiene_results: List[Dict],
    tooth_number_results: List[Dict]
) -> List[Dict]:
    """
    질병/위생/치아 번호 결과를 두 가지 겹침 비율 기반으로 매칭합니다.
    """
    final_matches = []
    
    all_detections = []
    
    # 특정 라벨만 매칭 대상으로 필터링
    allowed_disease_labels = ["충치 초기", "충치 중기", "충치 말기"]
    allowed_hygiene_labels = ["금니 (골드 크라운)", "은니 (메탈 크라운)", "아말감 충전재"]

    for det in disease_results:
        if det['label'] in allowed_disease_labels:
            all_detections.append({'type': 'disease', 'label': det['label'], 'confidence': det['confidence'], 'mask_array': det['mask_array']})

    for det in hygiene_results:
        if det['label'] in allowed_hygiene_labels:
            all_detections.append({'type': 'hygiene', 'label': det['label'], 'confidence': det['confidence'], 'mask_array': det['mask_array']})
    
    for tooth_info in tooth_number_results:
        tooth_number = tooth_info['tooth_number_fdi']
        tooth_bbox = [int(round(c)) for c in tooth_info['bbox']]
        
        # 각 치아 bbox에 대해 모든 질병/위생 탐지 결과를 순회
        for detection in all_detections:
            # 마스크를 원본 크기로 리사이징
            mask_orig = Image.fromarray(detection['mask_array']).resize(original_image_size, Image.NEAREST)
            mask_array_resized = np.array(mask_orig)

            # 두 가지 겹침 비율 계산
            ratio_mask_in_bbox, ratio_bbox_in_mask = get_overlap_ratios(mask_array_resized, tooth_bbox)
            
            # ⚠️ 원래의 임계값으로 롤백: 마스크가 bbox에 55% 이상 겹치거나, bbox가 마스크에 30% 이상 겹칠 때
            if (ratio_mask_in_bbox >= 0.55) or (ratio_bbox_in_mask >= 0.30):
                final_matches.append({
                    'tooth_number': tooth_number,
                    'category': detection['type'],
                    'label': detection['label'],
                    'confidence': detection['confidence'],
                    'overlap_ratio_mask_in_bbox': ratio_mask_in_bbox,
                    'overlap_ratio_bbox_in_mask': ratio_bbox_in_mask
                })

    # 동일한 치아-카테고리-라벨 조합에 대해 중복 제거 (가장 높은 confidence만 유지)
    unique_matches = {}
    for match in final_matches:
        key = (match['tooth_number'], match['category'], match['label'])
        if key not in unique_matches or match['confidence'] > unique_matches[key]['confidence']:
            unique_matches[key] = match

    return list(unique_matches.values())