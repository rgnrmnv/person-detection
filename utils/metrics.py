"""
Утилиты для расчета метрик
"""

import numpy as np
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class DetectionMetrics:
        
    
    total_frames: int = 0
    total_inference_time: float = 0.0
    total_preprocessing_time: float = 0.0
    total_postprocessing_time: float = 0.0
    
    
    total_detections: int = 0
    detections_per_frame: List[int] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    
   
    peak_gpu_memory: float = 0.0
    peak_cpu_memory: float = 0.0
    
    
    model_name: str = ""
    model_size_mb: float = 0.0
    input_size: tuple = (640, 640)
    
    def add_frame_result(
        self,
        detections: List[Dict],
        inference_time: float,
        preprocessing_time: float = 0.0,
        postprocessing_time: float = 0.0,
    ):
        """Добавление результатов с одного кадра"""
        self.total_frames += 1
        self.total_inference_time += inference_time
        self.total_preprocessing_time += preprocessing_time
        self.total_postprocessing_time += postprocessing_time
        
        num_detections = len(detections)
        self.total_detections += num_detections
        self.detections_per_frame.append(num_detections)
        
        for det in detections:
            if 'confidence' in det:
                self.confidence_scores.append(det['confidence'])
    
    @property
    def average_fps(self) -> float:
        if self.total_inference_time > 0:
            return self.total_frames / self.total_inference_time
        return 0.0
    
    @property
    def average_inference_time(self) -> float:
        if self.total_frames > 0:
            return (self.total_inference_time / self.total_frames) * 1000
        return 0.0
    
    @property
    def average_detections_per_frame(self) -> float:
        if self.total_frames > 0:
            return self.total_detections / self.total_frames
        return 0.0
    
    @property
    def average_confidence(self) -> float:
        if self.confidence_scores:
            return np.mean(self.confidence_scores)
        return 0.0
    
    @property
    def std_confidence(self) -> float:
        """Расчет стандартного отклонения уверенности"""
        if self.confidence_scores:
            return np.std(self.confidence_scores)
        return 0.0
    
    def to_dict(self) -> Dict:
        """Конвертация метрик в словарь"""
        return {
            'model_name': self.model_name,
            'model_size_mb': self.model_size_mb,
            'input_size': self.input_size,
            'performance': {
                'total_frames': self.total_frames,
                'average_fps': self.average_fps,
                'average_inference_time_ms': self.average_inference_time,
                'total_inference_time_s': self.total_inference_time,
                'total_preprocessing_time_s': self.total_preprocessing_time,
                'total_postprocessing_time_s': self.total_postprocessing_time,
            },
            'detections': {
                'total_detections': self.total_detections,
                'average_detections_per_frame': self.average_detections_per_frame,
                'min_detections_per_frame': min(self.detections_per_frame) if self.detections_per_frame else 0,
                'max_detections_per_frame': max(self.detections_per_frame) if self.detections_per_frame else 0,
                'std_detections_per_frame': np.std(self.detections_per_frame) if self.detections_per_frame else 0.0,
            },
            'confidence': {
                'average_confidence': self.average_confidence,
                'std_confidence': self.std_confidence,
                'min_confidence': min(self.confidence_scores) if self.confidence_scores else 0.0,
                'max_confidence': max(self.confidence_scores) if self.confidence_scores else 0.0,
            },
            'memory': {
                'peak_gpu_memory_mb': self.peak_gpu_memory,
                'peak_cpu_memory_mb': self.peak_cpu_memory,
            },
        }
    
    def summary(self) -> str:
        lines = [
            f"=== Метрики {self.model_name} ===",
            f"Обработано кадров: {self.total_frames}",
            f"Средний FPS: {self.average_fps:.2f}",
            f"Среднее время инференса: {self.average_inference_time:.2f} ms",
            f"Всего детекций: {self.total_detections}",
            f"Среднее детекций/кадр: {self.average_detections_per_frame:.2f}",
            f"Средняя уверенность: {self.average_confidence:.3f} ± {self.std_confidence:.3f}",
            f"Размер модели: {self.model_size_mb:.2f} MB",
        ]
        
        if self.peak_gpu_memory > 0:
            lines.append(f"Пиковая память GPU: {self.peak_gpu_memory:.2f} MB")
        
        return '\n'.join(lines)


def calculate_iou(box1: List[float], box2: List[float]) -> float:

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Вычисляем площадь пересечения
    intersect_xmin = max(x1_min, x2_min)
    intersect_ymin = max(y1_min, y2_min)
    intersect_xmax = min(x1_max, x2_max)
    intersect_ymax = min(y1_max, y2_max)
    
    if intersect_xmax < intersect_xmin or intersect_ymax < intersect_ymin:
        return 0.0
    
    intersect_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
    
    # Вычисляем площадь объединения
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersect_area
    
    if union_area == 0:
        return 0.0
    
    return intersect_area / union_area


def calculate_metrics(
    predictions: List[Dict],
    ground_truth: Optional[List[Dict]] = None,
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Вычисление различных метрик детекции
    
    Args:
        predictions (List[Dict]): Предсказанные детекции
        ground_truth (List[Dict], optional): Ground truth аннотации
        iou_threshold (float): Порог IoU для сопоставления
        
    Returns:
        Dict: Вычисленные метрики
    """
    metrics = {
        'num_predictions': len(predictions),
    }
    
    if predictions:
        confidences = [d['confidence'] for d in predictions if 'confidence' in d]
        if confidences:
            metrics['mean_confidence'] = np.mean(confidences)
            metrics['std_confidence'] = np.std(confidences)
            metrics['min_confidence'] = np.min(confidences)
            metrics['max_confidence'] = np.max(confidences)
    
    # Если есть ground truth, вычисляем precision/recall
    if ground_truth is not None:
        metrics['num_ground_truth'] = len(ground_truth)
        
        # Сопоставляем предсказания с ground truth
        matched_preds = set()
        matched_gts = set()
        
        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truth):
                if j in matched_gts:
                    continue
                
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou >= iou_threshold:
                    matched_preds.add(i)
                    matched_gts.add(j)
                    break
        
        true_positives = len(matched_preds)
        false_positives = len(predictions) - true_positives
        false_negatives = len(ground_truth) - len(matched_gts)
        
        precision = true_positives / len(predictions) if predictions else 0.0
        recall = true_positives / len(ground_truth) if ground_truth else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics.update({
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        })
    
    return metrics


class FPSCounter:
    """Счетчик FPS в реальном времени"""
    
    def __init__(self, window_size: int = 30):
        """
        Инициализация счетчика FPS
        
        Args:
            window_size (int): Размер окна скользящего среднего
        """
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self) -> float:
        """
        Обновление счетчика и возврат текущего FPS
        
        Returns:
            float: Текущий FPS
        """
        current_time = time.time()
        delta = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(delta)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_time if avg_time > 0 else 0.0
        
        return 0.0
    
    def reset(self):
        """Сброс счетчика FPS"""
        self.frame_times = []
        self.last_time = time.time()
