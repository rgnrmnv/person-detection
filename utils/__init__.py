"""
Утилиты для обработки видео, визуализации и метрик
"""

from .video_utils import VideoReader, VideoWriter
from .visualization import draw_detections, draw_info_panel, create_color_palette
from .metrics import calculate_metrics, DetectionMetrics, FPSCounter
from .nms import advanced_nms, apply_all_filters, remove_nested_detections
from .multiscale_detection import (
    detect_with_tiling,
    detect_multiscale,
    detect_with_upscaling,
    detect_with_small_object_boost
)

__all__ = [
    'VideoReader',
    'VideoWriter', 
    'draw_detections',
    'draw_info_panel',
    'create_color_palette',
    'calculate_metrics',
    'DetectionMetrics',
    'FPSCounter',
    'advanced_nms',
    'apply_all_filters',
    'remove_nested_detections',
    'detect_with_tiling',
    'detect_multiscale',
    'detect_with_upscaling',
    'detect_with_small_object_boost',
]
