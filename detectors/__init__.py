"""
Модуль детекторов 
"""

from .base_detector import BaseDetector
from .yolov8_detector import YOLOv8Detector
from .rtdetr_detector import RTDETRDetector

__all__ = ['BaseDetector', 'YOLOv8Detector', 'RTDETRDetector']


def get_detector(model_name, config_path=None, **kwargs):
    """
    Функция для создания детектора по имени
    
    Args:
        model_name (str): Название модели ('yolov8' или 'rtdetr')
        config_path (str, optional): Путь к конфигурационному файлу
        **kwargs: Дополнительные аргументы для детектора
        
    Returns:
        BaseDetector: Экземпляр запрошенного детектора
        
    Raises:
        ValueError: Если model_name не распознан
    """
    model_name = model_name.lower()
    
    if model_name == 'yolov8':
        return YOLOv8Detector(config_path=config_path, **kwargs)
    elif model_name == 'rtdetr':
        return RTDETRDetector(config_path=config_path, **kwargs)
    else:
        raise ValueError(
            f"Неизвестная модель: {model_name}. "
            f"Поддерживаемые модели: 'yolov8', 'rtdetr'"
        )
