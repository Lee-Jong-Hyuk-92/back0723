import numpy as np
from PIL import Image
from typing import List, Dict, Tuple

def get_overlap_ratio_mask_to_bbox(mask_array: np.ndarray, bbox: List[float]) -> float:
    """
    마스크 픽셀이 바운딩 박스에 포함되는 비율을 계산합니다.
    """
    if mask_array.ndim != 2:
        return 0.0

    # 바운딩 박스 좌표를 정수형으로 변환
    x1, y1, x2, y2 = [int(round(c)) for c in bbox]

    # 바운딩 박스 영역 내 마스크 픽셀 수 계산
    # 경계값 처리 (배열 범위를 벗어나지 않도록)
    y_min, y_max = max(0, y1), min(mask_array.shape[0], y2)
    x_min, x_max = max(0, x1), min(mask_array.shape[1], x2)
    
    if y_max <= y_min or x_max <= x_min:
        return 0.0

    bbox_roi = mask_array[y_min:y_max, x_min:x_max]
    overlap_pixels = np.count_nonzero(bbox_roi)

    # 전체 마스크 픽셀 수 계산
    total_mask_pixels = np.count_nonzero(mask_array)
    
    if total_mask_pixels == 0:
        return 0.0

    return overlap_pixels / total_mask_pixels

def combine_results(
    original_image_size: Tuple[int, int],
    disease_results: List[Dict],
    hygiene_results: List[Dict],
    tooth_number_results: List[Dict]
) -> List[Dict]:
    """
    질병/위생/치아 번호 결과를 마스크 포함 비율 기반으로 매칭합니다.
    """
    final_matches = []
    
    # 모든 탐지 결과를 하나의 리스트로 통합
    all_detections = []
    for det in disease_results:
        all_detections.append({'type': 'disease', 'label': det['label'], 'confidence': det['confidence'], 'mask_array': det['mask_array']})
    for det in hygiene_results:
        all_detections.append({'type': 'hygiene', 'label': det['label'], 'confidence': det['confidence'], 'mask_array': det['mask_array']})

    # 질병/위생 탐지 결과를 기준으로 루프
    for detection in all_detections:
        best_match_tooth = None
        max_ratio = 0.0
        
        # 마스크를 원본 크기로 리사이징
        mask_orig = Image.fromarray(detection['mask_array']).resize(original_image_size, Image.NEAREST)
        mask_array_resized = np.array(mask_orig)

        for tooth_info in tooth_number_results:
            tooth_bbox = tooth_info['bbox']
            
            # 마스크 포함 비율 계산
            overlap_ratio = get_overlap_ratio_mask_to_bbox(mask_array_resized, tooth_bbox)
            
            if overlap_ratio > max_ratio:
                max_ratio = overlap_ratio
                best_match_tooth = tooth_info
        
        # 55% 이상의 비율을 만족하는 경우에만 매칭
        if best_match_tooth and max_ratio >= 0.55:
            final_matches.append({
                'tooth_number': best_match_tooth['tooth_number_fdi'],
                'category': detection['type'],
                'label': detection['label'],
                'confidence': detection['confidence'],
                'overlap_ratio': max_ratio
            })

    return final_matches