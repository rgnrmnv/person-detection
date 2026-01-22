"""
Утилиты для чтения и записи видео
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class VideoReader:
   
    def __init__(self, video_path: str):
       
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Видео файл не найден: {video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {video_path}")
        
        # Получаем параметры видео
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0
        
        print(f"Видео загружено: {self.video_path.name}")
        print(f"  Разрешение: {self.width}x{self.height}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  Всего кадров: {self.frame_count}")
        print(f"  Длительность: {self.frame_count/self.fps:.2f}s")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
       
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
        return ret, frame
    
    def __iter__(self):
       
        return self
    
    def __next__(self) -> np.ndarray:
        ret, frame = self.read()
        if not ret:
            raise StopIteration
        return frame
    
    def get_properties(self) -> dict:
        return {
            'path': str(self.video_path),
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'duration': self.frame_count / self.fps if self.fps > 0 else 0,
        }
    
    def seek(self, frame_number: int) -> bool:
        """
        Переход к конкретному кадру
        """
        if 0 <= frame_number < self.frame_count:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
            return True
        return False
    
    def release(self):
        
        if self.cap is not None:
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def __del__(self):
        self.release()


class VideoWriter:
   
    CODEC_PREFERENCES = ['mp4v', 'avc1', 'XVID', 'MJPG']
    
    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        codec: Optional[str] = None,
    ):
        
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.frame_size = frame_size
        self.frame_count = 0
        
        # Determine codec
        if codec is None:
            codec = self._get_best_codec()
        
        self.fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            self.fourcc,
            self.fps,
            self.frame_size,
        )
        
        if not self.writer.isOpened():
            raise IOError(f"Failed to create video writer for: {output_path}")
        
        print(f"Video writer initialized:")
        print(f"  Output: {self.output_path}")
        print(f"  Codec: {codec}")
        print(f"  Resolution: {frame_size[0]}x{frame_size[1]}")
        print(f"  FPS: {fps:.2f}")
    
    def _get_best_codec(self) -> str:
        for codec in self.CODEC_PREFERENCES:
            test_writer = cv2.VideoWriter(
                'test.mp4',
                cv2.VideoWriter_fourcc(*codec),
                self.fps,
                self.frame_size,
            )
            if test_writer.isOpened():
                test_writer.release()
                Path('test.mp4').unlink(missing_ok=True)
                return codec
        
        
        print("Неудачная попытка. Используем mp4v.")
        return 'mp4v'
    
    def write(self, frame: np.ndarray):
        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            # Resize if necessary
            frame = cv2.resize(frame, self.frame_size)
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def release(self):
        if self.writer is not None:
            self.writer.release()
            print(f"Видео сохранено: {self.output_path} ({self.frame_count} кадров)")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def __del__(self):
        self.release()


def get_video_info(video_path: str) -> dict:
    with VideoReader(video_path) as reader:
        return reader.get_properties()
