# Упрощенная реализация для отслеживания людей

import numpy as np
from typing import List, Dict
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment


class STrack:
    """Один трек объекта"""
    
    _count = 0  # Глобальный счетчик ID треков
    
    def __init__(self, bbox, score):
        """
        Инициализация трека
        
        Args:
            bbox (list): Bounding box [x1, y1, x2, y2]
            score (float): Уверенность детекции
        """
        self.bbox = np.array(bbox)
        self.score = score
        
        # Состояние трека
        self.track_id = STrack._count
        STrack._count += 1
        
        self.state = 'new'  
        self.is_activated = False
        
        # История отслеживания
        self.frame_id = 0
        self.tracklet_len = 0
        self.time_since_update = 0
        
        # Фильтр Калмана (очень упрощенный)
        self.velocity = np.zeros(4)
    
    def activate(self, frame_id):
        """Активация трека"""
        self.frame_id = frame_id
        self.tracklet_len = 0
        self.state = 'tracked'
        self.is_activated = True
    
    def re_activate(self, new_track, frame_id):
        """Повторная активация потерянного трека"""
        self.bbox = new_track.bbox
        self.score = new_track.score
        self.tracklet_len = 0
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id
        self.time_since_update = 0
    
    def update(self, new_track, frame_id):
        """Обновление трека новой детекцией"""
        # Обновляем скорость (простая модель движения)
        self.velocity = new_track.bbox - self.bbox
        
        self.bbox = new_track.bbox
        self.score = new_track.score
        self.tracklet_len += 1
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id
        self.time_since_update = 0
    
    def predict(self):
        """Предсказание следующей позиции используя простую модель движения"""
        if self.state != 'tracked':
            self.velocity *= 0  # Сброс скорости если не отслеживается
        
        # Предсказание со скоростью
        self.bbox += self.velocity
        
        # Clip to ensure valid bbox
        self.bbox[0] = max(0, self.bbox[0])  # x1
        self.bbox[1] = max(0, self.bbox[1])  # y1
    
    def mark_lost(self):
        """Отметить трек как потерянный"""
        self.state = 'lost'
        self.time_since_update += 1
    
    def mark_removed(self):
        """Отметить трек как удаленный"""
        self.state = 'removed'
    
    @property
    def tlwh(self):
        """Получить bbox в формате [x, y, w, h]"""
        ret = self.bbox.copy()
        ret[2:] -= ret[:2]  
        return ret
    
    @property
    def tlbr(self):
        """Получить bbox в формате [x1, y1, x2, y2]"""
        return self.bbox.copy()
    
    def __repr__(self):
        return f"STrack(id={self.track_id}, state={self.state}, score={self.score:.2f})"


class ByteTracker:
    """
    Реализация ByteTrack трекера
    Простое и эффективное отслеживание множества объектов
    """
    
    def __init__(
        self,
        track_threshold: float = 0.5,
        track_buffer: int = 30,
        match_threshold: float = 0.8,
        min_box_area: float = 10,
    ):
        """
        Инициализация ByteTracker
        
        Args:
            track_threshold (float): Порог уверенности для высокоскоринговых детекций
            track_buffer (int): Количество кадров для хранения потерянных треков
            match_threshold (float): Порог IoU для сопоставления
            min_box_area (float): Минимальная площадь bbox
        """
        self.track_threshold = track_threshold
        self.track_buffer = track_buffer
        self.match_threshold = match_threshold
        self.min_box_area = min_box_area
        
        # Управление треками
        self.tracked_tracks = []  
        self.lost_tracks = []     
        self.removed_tracks = []  
        
        self.frame_id = 0
        
        # Сброс глобального счетчика ID треков
        STrack._count = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Обновление трекера новыми детекциями
        
        Args:
            detections (List[Dict]): Список детекций с 'bbox' и 'confidence'
            
        Returns:
            List[Dict]: Детекции с добавленным полем 'track_id'
        """
        self.frame_id += 1
        
        # Конвертируем детекции в объекты STrack
        det_tracks = []
        for det in detections:
            bbox = det['bbox']
            score = det['confidence']
            
            # Фильтруем по площади
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w * h < self.min_box_area:
                continue
            
            det_tracks.append(STrack(bbox, score))
        
        # Разделяем высоко- и низкоскоринговые детекции (стратегия ByteTrack)
        high_det_tracks = [t for t in det_tracks if t.score >= self.track_threshold]
        low_det_tracks = [t for t in det_tracks if t.score < self.track_threshold]
        
        # Предсказываем текущую позицию треков
        for track in self.tracked_tracks:
            track.predict()
        
        # Первая ассоциация: сопоставляем высокоскоринговые детекции с отслеживаемыми треками
        matched, unmatched_tracks, unmatched_dets = self._match(
            self.tracked_tracks, high_det_tracks, self.match_threshold
        )
        
        # Обновляем сопоставленные треки
        for track_idx, det_idx in matched:
            track = self.tracked_tracks[track_idx]
            det = high_det_tracks[det_idx]
            track.update(det, self.frame_id)
        
        # Вторая ассоциация: сопоставляем низкоскоринговые детекции с несопоставленными треками
        unmatched_track_objects = [self.tracked_tracks[i] for i in unmatched_tracks]
        matched2, unmatched_tracks2, unmatched_dets2 = self._match(
            unmatched_track_objects, low_det_tracks, 0.5  # Меньший порог
        )
        
        # Обновляем треки из второй ассоциации
        for track_idx, det_idx in matched2:
            track = unmatched_track_objects[track_idx]
            det = low_det_tracks[det_idx]
            track.update(det, self.frame_id)
        
        # Отмечаем несопоставленные треки как потерянные
        for i in unmatched_tracks2:
            track = unmatched_track_objects[i]
            track.mark_lost()
        
        # Инициализируем новые треки из несопоставленных высокоскоринговых детекций
        for i in unmatched_dets:
            track = high_det_tracks[i]
            if track.score >= self.track_threshold:
                track.activate(self.frame_id)
                self.tracked_tracks.append(track)
        
        # Обновляем состояние
        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == 'tracked']
        
        # Объединяем отслеживаемые и потерянные треки
        self.tracked_tracks, self.lost_tracks = self._merge_tracks(
            self.tracked_tracks, self.lost_tracks
        )
        
        # Удаляем старые потерянные треки
        self.lost_tracks = [
            t for t in self.lost_tracks 
            if self.frame_id - t.frame_id <= self.track_buffer
        ]
        
        # Подготавливаем выход: добавляем track ID к оригинальным детекциям
        output_detections = []
        active_tracks = [t for t in self.tracked_tracks if t.is_activated]
        
        for track in active_tracks:
            # Находим соответствующую детекцию
            for det in detections:
                det_bbox = np.array(det['bbox'])
                if np.allclose(track.tlbr, det_bbox, atol=1):
                    det_copy = det.copy()
                    det_copy['track_id'] = track.track_id
                    output_detections.append(det_copy)
                    break
        
        return output_detections
    
    def _match(self, tracks, detections, threshold):
        """
        Сопоставление треков с детекциями по IoU
        
        Args:
            tracks (List[STrack]): Список треков
            detections (List[STrack]): Список детекций
            threshold (float): Порог IoU
            
        Returns:
            Tuple: (сопоставленные пары, несопоставленные треки, несопоставленные детекции)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Вычисляем матрицу стоимости IoU
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = self._iou(track.tlbr, det.tlbr)
        
        # Преобразуем IoU в стоимость (1 - IoU)
        cost_matrix = 1 - cost_matrix
        
        # Венгерский алгоритм для оптимального сопоставления
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Фильтруем сопоставления по порогу
        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            iou = 1 - cost_matrix[track_idx, det_idx]
            if iou >= threshold:
                matched.append((track_idx, det_idx))
                if track_idx in unmatched_tracks:
                    unmatched_tracks.remove(track_idx)
                if det_idx in unmatched_dets:
                    unmatched_dets.remove(det_idx)
        
        return matched, unmatched_tracks, unmatched_dets
    
    def _iou(self, bbox1, bbox2):
        """
        Вычисление IoU между двумя bbox
        
        Args:
            bbox1 (np.array): Первый bbox [x1, y1, x2, y2]
            bbox2 (np.array): Второй bbox [x1, y1, x2, y2]
            
        Returns:
            float: Значение IoU
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _merge_tracks(self, tracked, lost):
        """Объединение отслеживаемых и потерянных треков, удаление дубликатов"""
        return tracked, lost
    
    def reset(self):
        """Сброс трекера"""
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        STrack._count = 0
