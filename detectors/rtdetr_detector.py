"""
Детектор RT-DETR
"""

import torch
import numpy as np
from pathlib import Path
from ultralytics import RTDETR
from .base_detector import BaseDetector


class RTDETRDetector(BaseDetector):
        
    # Названия классов COCO (те же что и у YOLO)
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
        Инициализация детектора RT-DETR
        
        Args:
            config_path (str, optional): Путь к файлу конфигурации
            **kwargs: Дополнительные параметры
        """
        # Устанавливаем путь к конфигу по умолчанию если не указан
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'config' / 'rtdetr_config.yaml'
        
        super().__init__(config_path, **kwargs)
        self.load_model()
    
    def load_model(self):
        """Загрузка модели RT-DETR"""
        model_config = self.config.get('model', {})
        weights_name = model_config.get('weights', 'rtdetr-l.pt')
        
        print(f"Загрузка модели RT-DETR: {weights_name}")
        
        # Ultralytics автоматически скачает веса если их нет
        self.model = RTDETR(weights_name)
        
        # Перенос на устройство
        if self.device.startswith('cuda'):
            self.model.to(self.device)
        
        # Информация о модели
        self.model_info = {
            'name': 'RT-DETR',
            'variant': model_config.get('variant', 'rtdetr-l'),
            'input_size': tuple(model_config.get('input_size', [640, 640])),
        }
        
        print(f"Модель RT-DETR успешно загружена на {self.device}")
        print(f"Вариант модели: {self.model_info['variant']}")
    
    def preprocess(self, frame):
        """
        Предобработка кадра для RT-DETR
        
        Args:
            frame (numpy.ndarray): Входной кадр (BGR)
            
        Returns:
            numpy.ndarray: Предобработанный кадр
        """
        # RT-DETR выполняет предобработку внутренне (как и YOLOv8)
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
        target_class = det_config.get('target_class', 0)  # person (человек)
        
        # Инференс
        start_time = time.time()
        
        # RT-DETR не использует NMS (встроен в архитектуру)
        # Примечание: параметр iou не используется в RT-DETR
        results = self.model.predict(
            frame,
            conf=conf_threshold,
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
        Постобработка предсказаний RT-DETR
        
        Args:
            predictions: Объект Results от RT-DETR
            original_shape (tuple): Исходный размер кадра
            
        Returns:
            list: Список словарей с детекциями
        """
        detections = []
        
        # RT-DETR возвращает объекты Results (та же структура что и YOLOv8)
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
    
    def export_onnx(self, output_path='weights/rtdetr.onnx'):
        """
        Экспорт модели в формат ONNX
        
        Args:
            output_path (str): Путь для сохранения ONNX модели
        """
        print(f"Экспорт RT-DETR в ONNX: {output_path}")
        
        export_config = self.config.get('export', {}).get('onnx', {})
        
        self.model.export(
            format='onnx',
            opset=export_config.get('opset', 12),
            simplify=export_config.get('simplify', True),
            dynamic=export_config.get('dynamic', False),
        )
        
        print(f"Модель успешно экспортирована в {output_path}")
    
    def get_model_info(self):
        """Возвращает информацию о модели"""
        return {
            'name': 'RT-DETR',
            'type': 'transformer-based-detector',
            'architecture': 'DETR',
            'framework': 'Ultralytics',
            'device': str(self.device),
            'nms': 'Not required',
            **self.model_info,
        }
