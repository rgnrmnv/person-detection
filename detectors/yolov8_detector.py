"""
Детектор YOLOv8
"""

import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from .base_detector import BaseDetector


class YOLOv8Detector(BaseDetector):  
    # Названия классов COCO
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    def __init__(self, config_path=None, **kwargs):
        """
        Инициализация детектора YOLOv8
        
        Args:
            config_path (str, optional): Путь к файлу конфигурации
            **kwargs: Дополнительные параметры
        """
        # Устанавливаем путь к конфигу по умолчанию если не указан
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'yolov8_config.yaml'
        
        super().__init__(config_path, **kwargs)
        self.load_model()
    
    def load_model(self):
        """Загрузка модели YOLOv8"""
        model_config = self.config.get('model', {})
        weights_name = model_config.get('weights', 'yolov8m.pt')
        
        print(f"Загрузка модели YOLOv8: {weights_name}")
        
        # Ultralytics автоматически скачает веса если их нет
        self.model = YOLO(weights_name)
        
        # Перенос на устройство
        if self.device.startswith('cuda'):
            self.model.to(self.device)
        
        # Информация о модели
        self.model_info = {
            'name': 'YOLOv8',
            'variant': model_config.get('variant', 'yolov8m'),
            'input_size': tuple(model_config.get('input_size', [640, 640])),
        }
        
        print(f"Модель YOLOv8 успешно загружена на {self.device}")
        print(f"Вариант модели: {self.model_info['variant']}")
    
    def preprocess(self, frame):
        """
        Предобработка кадра для YOLOv8
        
        Args:
            frame (numpy.ndarray): Входной кадр (BGR)
            
        Returns:
            numpy.ndarray: Предобработанный кадр
        """
        # YOLOv8 выполняет предобработку внутренне
        return frame
    
    def detect(self, frame):
        """
        
        Args:
            frame (numpy.ndarray): Входной кадр (BGR)
            
        Returns:
            list: Список детекций
        """
        import time
        
        # Получаем параметры детекции
        det_config = self.config.get('detection', {})
        conf_threshold = det_config.get('confidence_threshold', 0.25)
        iou_threshold = det_config.get('iou_threshold', 0.45)
        target_class = det_config.get('target_class', 0)  # person (человек)
        
        # Инференс
        start_time = time.time()
        
        results = self.model.predict(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=[target_class],  # Детектируем только людей
            verbose=False,
            device=self.device,
        )
        
        inference_time = time.time() - start_time
        
        # Обновляем статистику
        self.total_inference_time += inference_time
        self.total_frames += 1
        
        # Парсим результаты
        detections = self.postprocess(results, frame.shape)
        self.total_detections += len(detections)
        
        return detections
    
    def postprocess(self, predictions, original_shape):
        """
        Постобработка предсказаний YOLOv8
        
        Args:
            predictions: Объект Results от YOLOv8
            original_shape (tuple): Исходный размер кадра
            
        Returns:
            list: Список словарей с детекциями
        """
        detections = []
        
        # YOLOv8 возвращает объекты Results
        for result in predictions:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            # Извлекаем данные детекций
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                detection = {
                    'bbox': box.tolist(),
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': self.COCO_CLASSES[cls_id] if cls_id < len(self.COCO_CLASSES) else 'unknown',
                }
                
                detections.append(detection)
        
        return detections
    
    def export_onnx(self, output_path='weights/yolov8.onnx'):
        """
        Экспорт модели в формат ONNX
        
        Args:
            output_path (str): Путь для сохранения ONNX модели
        """
        print(f"Экспорт YOLOv8 в ONNX: {output_path}")
        
        export_config = self.config.get('export', {}).get('onnx', {})
        
        self.model.export(
            format='onnx',
            opset=export_config.get('opset', 12),
            simplify=export_config.get('simplify', True),
            dynamic=export_config.get('dynamic', False),
        )
        
        print(f"Модель успешно экспортирована в {output_path}")
    
    def export_tensorrt(self, output_path='weights/yolov8.engine'):
        """
        Экспорт модели в формат TensorRT
        
        Args:
            output_path (str): Путь для сохранения TensorRT engine
        """
        print(f"Экспорт YOLOv8 в TensorRT: {output_path}")
        
        export_config = self.config.get('export', {}).get('tensorrt', {})
        
        self.model.export(
            format='engine',
            half=export_config.get('precision', 'fp16') == 'fp16',
            workspace=export_config.get('workspace', 4),
        )
        
        print(f"Модель успешно экспортирована в {output_path}")
    
    def get_model_info(self):
        """Возвращает информацию о модели"""
        return {
            'name': 'YOLOv8',
            'type': 'one-stage-detector',
            'architecture': 'CNN-based',
            'framework': 'Ultralytics',
            'device': str(self.device),
            **self.model_info,
        }
