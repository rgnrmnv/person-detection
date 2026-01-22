# Детекция маленький объектов

import cv2
import numpy as np
from typing import List, Dict, Tuple
from .nms import advanced_nms


def tile_image(image: np.ndarray, tile_size: int = 640, overlap: float = 0.2) -> List[Dict]:
  
    h, w = image.shape[:2]
    stride = int(tile_size * (1 - overlap))
    
    tiles = []
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Границы тайла
            x1 = x
            y1 = y
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            
            # Извлекаем тайл
            tile = image[y1:y2, x1:x2]
            
            # Если тайл меньше tile_size, дополняем
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            
            tiles.append({
                'tile': tile,
                'offset': (x1, y1),
                'original_size': (x2 - x1, y2 - y1)
            })
    
    return tiles


def adjust_detections_to_original(
    detections: List[Dict],
    offset: Tuple[int, int]
) -> List[Dict]:
    """
    Корректирует координаты детекций
    
    Args:
        detections: Детекции на тайле
        offset: Смещение тайла (x, y)
    
    Returns:
        Детекции с скорректированными координатами
    """
    adjusted = []
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        offset_x, offset_y = offset
        
        adjusted.append({
            'bbox': [
                x1 + offset_x,
                y1 + offset_y,
                x2 + offset_x,
                y2 + offset_y
            ],
            'confidence': det['confidence'],
            'class_id': det.get('class_id', 0),
            'class_name': det.get('class_name', 'person')
        })
    
    return adjusted


def detect_with_tiling(
    image: np.ndarray,
    detector,
    tile_size: int = 640,
    overlap: float = 0.3,
    conf_threshold: float = 0.35,
    nms_iou: float = 0.5
) -> List[Dict]:
    """
      
    Args:
        image: Исходное изображение
        detector: Детектор объектов
        tile_size: Размер тайла
        overlap: Перекрытие между тайлами
        conf_threshold: Порог уверенности
        nms_iou: Порог IoU для NMS
    
    Returns:
        Список детекций
    """
    # Разбиваем на тайлы
    tiles = tile_image(image, tile_size, overlap)
    
    all_detections = []
    
    # Детекция на каждом тайле
    for tile_info in tiles:
        tile = tile_info['tile']
        offset = tile_info['offset']
        
        # Запускаем детекцию
        tile_detections = detector.detect(tile)
        
        # Фильтруем по уверенности
        tile_detections = [
            d for d in tile_detections 
            if d['confidence'] >= conf_threshold
        ]
        
        # Корректируем координаты
        adjusted = adjust_detections_to_original(tile_detections, offset)
        all_detections.extend(adjusted)
    
    # Применяем NMS ко всем детекциям
    if len(all_detections) > 0:
        all_detections = advanced_nms(all_detections, nms_iou)
    
    return all_detections


def detect_multiscale(
    image: np.ndarray,
    detector,
    scales: List[float] = [1.0, 1.5, 2.0],
    conf_threshold: float = 0.35,
    nms_iou: float = 0.5
) -> List[Dict]:
    """
     
    Обрабатывает изображение в разных масштабах для лучшей детекции
    маленьких и больших объектов
    
    Args:
        image: Исходное изображение
        detector: Детектор объектов
        scales: Список масштабов (1.0 = оригинал)
        conf_threshold: Порог уверенности
        nms_iou: Порог IoU для NMS
    
    Returns:
        Список детекций
    """
    h, w = image.shape[:2]
    all_detections = []
    
    for scale in scales:
        # Изменяем размер изображения
        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            scaled_image = cv2.resize(image, (new_w, new_h))
        else:
            scaled_image = image
        
        # Детекция на масштабированном изображении
        detections = detector.detect(scaled_image)
        
        # Фильтруем по уверенности
        detections = [
            d for d in detections 
            if d['confidence'] >= conf_threshold
        ]
        
        # Корректируем координаты обратно к оригинальному размеру
        if scale != 1.0:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                det['bbox'] = [
                    int(x1 / scale),
                    int(y1 / scale),
                    int(x2 / scale),
                    int(y2 / scale)
                ]
        
        all_detections.extend(detections)
    
    # Применяем NMS ко всем детекциям
    if len(all_detections) > 0:
        all_detections = advanced_nms(all_detections, nms_iou)
    
    return all_detections


def detect_with_upscaling(
    image: np.ndarray,
    detector,
    target_size: int = 1280,
    conf_threshold: float = 0.35
) -> List[Dict]:
    """
    Детекция с предварительным увеличением разрешения
    
    Args:
        image: Исходное изображение
        detector: Детектор объектов
        target_size: Целевой размер (по большей стороне)
        conf_threshold: Порог уверенности
    
    Returns:
        Список детекций
    """
    h, w = image.shape[:2]
    
    # Вычисляем масштаб
    max_side = max(h, w)
    scale = target_size / max_side
    
    # Увеличиваем изображение
    if scale > 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        upscaled = image
        scale = 1.0
    
    # Детекция
    detections = detector.detect(upscaled)
    
    # Фильтруем по уверенности
    detections = [
        d for d in detections 
        if d['confidence'] >= conf_threshold
    ]
    
    # Возвращаем координаты к оригинальному размеру
    if scale != 1.0:
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            det['bbox'] = [
                int(x1 / scale),
                int(y1 / scale),
                int(x2 / scale),
                int(y2 / scale)
            ]
    
    return detections


def detect_with_small_object_boost(
    image: np.ndarray,
    detector,
    method: str = 'tiling',
    **kwargs
) -> List[Dict]:
   
    if method == 'tiling':
        return detect_with_tiling(
            image, detector,
            tile_size=kwargs.get('tile_size', 640),
            overlap=kwargs.get('overlap', 0.3),
            conf_threshold=kwargs.get('conf_threshold', 0.35),
            nms_iou=kwargs.get('nms_iou', 0.5)
        )
    
    elif method == 'multiscale':
        return detect_multiscale(
            image, detector,
            scales=kwargs.get('scales', [1.0, 1.5, 2.0]),
            conf_threshold=kwargs.get('conf_threshold', 0.35),
            nms_iou=kwargs.get('nms_iou', 0.5)
        )
    
    elif method == 'upscaling':
        return detect_with_upscaling(
            image, detector,
            target_size=kwargs.get('target_size', 1280),
            conf_threshold=kwargs.get('conf_threshold', 0.35)
        )
    
    else:
        # Обычная детекция
        return detector.detect(image)
