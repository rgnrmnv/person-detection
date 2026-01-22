"""
Абстрактный класс для детекторов
"""

from abc import ABC, abstractmethod
from pathlib import Path
import time
import yaml
import numpy as np
import torch


class BaseDetector(ABC):
        
    def __init__(self, config_path=None, **kwargs):
        """
        Инициализация детектора
        
        Args:
            config_path (str, optional): Путь к YAML файлу конфигурации
            **kwargs: Дополнительные параметры для переопределения конфига
        """
        self.config = self._load_config(config_path)
        
        # Переопределяем конфиг параметрами из kwargs
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
        
        self.model = None
        self.device = self._setup_device()
        self.model_name = self.config.get('model', {}).get('name', 'unknown')
        
        # Статистика
        self.total_detections = 0
        self.total_frames = 0
        self.total_inference_time = 0.0
        
    def _load_config(self, config_path):
        """Загружает конфигурацию из YAML файла"""
        if config_path is None:
            return self._get_default_config()
        
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"Предупреждение: Файл конфигурации {config_path} не найден. Используем значения по умолчанию.")
            return self._get_default_config()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _get_default_config(self):
        """Возвращает конфигурацию по умолчанию"""
        return {
            'model': {
                'name': 'base',
                'backend': 'pytorch',
            },
            'detection': {
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_detections': 300,
                'target_class': 0,  # person (класс человек)
            },
            'optimization': {
                'device': 'auto',
                'half_precision': False,
                'batch_size': 1,
            },
        }
    
    def _setup_device(self):
        """Настройка устройства для вычислений (CPU/CUDA)"""
        device_config = self.config.get('optimization', {}).get('device', 'auto')
        
        if device_config == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device_config
        
        print(f"Используется устройство: {device}")
        if device.startswith('cuda') and torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA версия: {torch.version.cuda}")
        
        return device
    
    @abstractmethod
    def load_model(self):
        """Загружает и инициализирует модель детекции"""
        pass
    
    @abstractmethod
    def detect(self, frame):
        """
        Выполняет детекцию на одном кадре
        
        Args:
            frame (numpy.ndarray): Входное изображение в BGR формате
            
        Returns:
            list: Список детекций, каждая как словарь с ключами:
                  - bbox: [x1, y1, x2, y2]
                  - confidence: float
                  - class_id: int
                  - class_name: str
        """
        pass
    
    @abstractmethod
    def preprocess(self, frame):
        """
        Предобработка кадра перед детекцией
        
        Args:
            frame (numpy.ndarray): Входное изображение
            
        Returns:
            Any: Предобработанный вход для модели
        """
        pass
    
    @abstractmethod
    def postprocess(self, predictions, original_shape):
        """
        Постобработка предсказаний модели
        
        Args:
            predictions: Сырой выход модели
            original_shape (tuple): Исходный размер изображения (H, W, C)
            
        Returns:
            list: Список обработанных детекций
        """
        pass
    
    def detect_batch(self, frames):
        """
        Выполняет детекцию на пакете кадров
        
        Args:
            frames (list): Список кадров
            
        Returns:
            list: Список результатов детекции для каждого кадра
        """
        results = []
        for frame in frames:
            detections = self.detect(frame)
            results.append(detections)
        return results
    
    def filter_by_class(self, detections, taЗrget_class=0):
        """
        Фильтрация детекций по ID класса
        
        Args:
            detections (list): Список всех детекций
            target_class (int): ID целевого класса (0 для person в COCO)
            
        Returns:
            list: Отфильтрованные детекции
        """
        return [d for d in detections if d['class_id'] == target_class]
    
    def filter_by_confidence(self, detections, threshold=None):
        """
        Фильтрация детекций по порогу 
        
        Args:
            detections (list): Список детекций
            threshold (float, optional): Порог 
            
        Returns:
            list: Отфильтрованные детекции
        """
        if threshold is None:
            threshold = self.config.get('detection', {}).get('confidence_threshold', 0.25)
        
        return [d for d in detections if d['confidence'] >= threshold]
    
    def get_statistics(self):
        """Возвращает статистику детекций"""
        avg_fps = self.total_frames / self.total_inference_time if self.total_inference_time > 0 else 0
        avg_detections = self.total_detections / self.total_frames if self.total_frames > 0 else 0
        
        return {
            'model_name': self.model_name,
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'total_inference_time': self.total_inference_time,
            'average_fps': avg_fps,
            'average_detections_per_frame': avg_detections,
        }
    
    def reset_statistics(self):
        """Сбрасывает статистику детекций"""
        self.total_detections = 0
        self.total_frames = 0
        self.total_inference_time = 0.0
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device})"
