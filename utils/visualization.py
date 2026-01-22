"""
Визуализация отрисовки детекций на кадрах
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import colorsys


def create_color_palette(n_colors: int = 80) -> List[Tuple[int, int, int]]:
    """
    Список BGR цветов
    """
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        saturation = 0.9
        value = 0.9
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    
    return colors



COLOR_PALETTE = create_color_palette(80)


def get_color_for_id(obj_id: int) -> Tuple[int, int, int]:
    return COLOR_PALETTE[obj_id % len(COLOR_PALETTE)]


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict],
    show_confidence: bool = True,
    show_class: bool = True,
    show_id: bool = False,
    bbox_thickness: int = 2,
    text_scale: float = 0.6,
    text_thickness: int = 2,
    color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    frame = frame.copy()
    
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Выбираем цвет
        if color is not None:
            box_color = color
        elif 'track_id' in det and show_id:
            box_color = get_color_for_id(det['track_id'])
        else:
            box_color = (0, 255, 0)
        
        # Рисуем bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, bbox_thickness)
        
        # Подготавливаем текст метки
        label_parts = []
        
        if show_id and 'track_id' in det:
            label_parts.append(f"ID:{det['track_id']}")
        
        if show_class and 'class_name' in det:
            label_parts.append(det['class_name'])
        
        if show_confidence and 'confidence' in det:
            label_parts.append(f"{det['confidence']:.2f}")
        
        if label_parts:
            label = ' '.join(label_parts)
            
            # Получаем размер текста
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                text_thickness
            )
            
            # Рисуем фон метки
            label_y1 = max(y1 - text_height - baseline - 5, 0)
            label_y2 = y1
            cv2.rectangle(
                frame,
                (x1, label_y1),
                (x1 + text_width + 5, label_y2),
                box_color,
                -1
            )
            
            # Рисуем текст метки
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA
            )
    
    return frame


def draw_info_panel(
    frame: np.ndarray,
    model_name: str,
    fps: float,
    detection_count: int,
    frame_number: int,
    total_frames: int,
) -> np.ndarray:
    
    frame = frame.copy()
    
    # Конфигурация панели
    panel_height = 100
    panel_color = (0, 0, 0)
    text_color = (255, 255, 255)
    text_scale = 0.6
    text_thickness = 2
    line_height = 25
    
    # Рисуем полупрозрачную панель
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), panel_color, -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Рисуем текстовую информацию
    y_offset = 25
    
    # Название модели
    cv2.putText(
        frame,
        f"Model: {model_name}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA
    )
    
    # FPS
    y_offset += line_height
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA
    )
    
    # Детекции
    y_offset += line_height
    cv2.putText(
        frame,
        f"Detections: {detection_count}",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA
    )
    
    # Прогресс кадров
    progress_text = f"Frame: {frame_number}/{total_frames}"
    text_width = cv2.getTextSize(
        progress_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        text_thickness
    )[0][0]
    
    cv2.putText(
        frame,
        progress_text,
        (frame.shape[1] - text_width - 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA
    )
    
    return frame


def draw_fps_counter(
    frame: np.ndarray,
    fps: float,
    position: str = 'top-right',
) -> np.ndarray:
    
    frame = frame.copy()
    
    text = f"FPS: {fps:.1f}"
    text_scale = 0.7
    text_thickness = 2
    
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        text_thickness
    )
    
    # Вычисляем позицию
    h, w = frame.shape[:2]
    margin = 10
    
    if position == 'top-left':
        x, y = margin, text_height + margin
    elif position == 'top-right':
        x, y = w - text_width - margin, text_height + margin
    elif position == 'bottom-left':
        x, y = margin, h - margin
    else:
        x, y = w - text_width - margin, h - margin
    
    # Рисуем фон
    cv2.rectangle(
        frame,
        (x - 5, y - text_height - 5),
        (x + text_width + 5, y + baseline + 5),
        (0, 0, 0),
        -1
    )
    
    # Рисуем текст
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (0, 255, 0),
        text_thickness,
        cv2.LINE_AA
    )
    
    return frame
