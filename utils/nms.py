"""
NMS для удаления дубликатов детекций
"""

import numpy as np
from typing import List, Dict


def calculate_iou(box1, box2):
   
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Вычисляем координаты пересечения
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Площадь пересечения
    if x2_i < x1_i or y2_i < y1_i:
        intersection = 0.0
    else:
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Площади bbox'ов
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # IoU
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def advanced_nms(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    
    if len(detections) == 0:
        return []
    
    # Сортируем по уверенности (от большей к меньшей)
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # Список отобранных детекций
    keep = []
    
    for det in sorted_dets:
        
        is_duplicate = False
        
        for kept_det in keep:
            iou = calculate_iou(det['bbox'], kept_det['bbox'])
            
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        # Если не дубликат, добавляем в список
        if not is_duplicate:
            keep.append(det)
    
    return keep


def merge_close_detections(
    detections: List[Dict],
    iou_threshold: float = 0.3,
    confidence_threshold: float = 0.4
) -> List[Dict]:
    
    if len(detections) == 0:
        return []
    
    # Разделяем на уверенные и неуверенные
    confident = [d for d in detections if d['confidence'] >= confidence_threshold]
    uncertain = [d for d in detections if d['confidence'] < confidence_threshold]
    
    
    merged = confident.copy()
    
    for unc_det in uncertain:
        max_iou = 0
        best_match_idx = -1
        
        
        for idx, conf_det in enumerate(merged):
            iou = calculate_iou(unc_det['bbox'], conf_det['bbox'])
            if iou > max_iou:
                max_iou = iou
                best_match_idx = idx
        
        
        if max_iou > iou_threshold:
            continue
        else:
            
            merged.append(unc_det)
    
    return merged


def remove_nested_detections(detections: List[Dict], containment_threshold: float = 0.8) -> List[Dict]:
    """
    Удаляет детекции, полностью вложенные в другие (большие) детекции
    
    """
    if len(detections) <= 1:
        return detections
    
    sorted_dets = sorted(
        detections,
        key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]),
        reverse=True
    )
    
    keep = []
    
    for i, det in enumerate(sorted_dets):
        is_nested = False
        
        for larger_det in sorted_dets[:i]:
            
            x1, y1, x2, y2 = det['bbox']
            x1_l, y1_l, x2_l, y2_l = larger_det['bbox']
            
            # Пересечение
            x1_i = max(x1, x1_l)
            y1_i = max(y1, y1_l)
            x2_i = min(x2, x2_l)
            y2_i = min(y2, y2_l)
            
            if x2_i >= x1_i and y2_i >= y1_i:
                intersection = (x2_i - x1_i) * (y2_i - y1_i)
                det_area = (x2 - x1) * (y2 - y1)
                
                
                if intersection / det_area > containment_threshold:
                    is_nested = True
                    break
        
        if not is_nested:
            keep.append(det)
    
    return keep


def apply_all_filters(
    detections: List[Dict],
    nms_iou: float = 0.5,
    merge_iou: float = 0.3,
    confidence_threshold: float = 0.4,
    containment_threshold: float = 0.8
) -> List[Dict]:
   
    if len(detections) == 0:
        return []
    
    
    filtered = remove_nested_detections(detections, containment_threshold)
    
    filtered = merge_close_detections(filtered, merge_iou, confidence_threshold)
    
    filtered = advanced_nms(filtered, nms_iou)
    
    return filtered
