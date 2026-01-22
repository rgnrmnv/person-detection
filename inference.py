"""
Основной скрипт для детекции людей на видео.
"""

import argparse
import sys
import time
from pathlib import Path
from tqdm import tqdm

# Добавляем корень проекта в путь, чтобы импорты работали корректно
sys.path.append(str(Path(__file__).parent))

from detectors import get_detector
from tracking import ByteTracker
from utils import VideoReader, VideoWriter, draw_detections, draw_info_panel, FPSCounter
from utils.metrics import DetectionMetrics
from utils.multiscale_detection import detect_with_tiling


def parse_args():
    parser = argparse.ArgumentParser(
        description='Детекция людей на видео',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['yolov8', 'rtdetr'],
        help='Модель для детекции'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='crowd.mp4',
        help='Путь к входному видео'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Путь к выходному видео (по умолчанию: results/{model}_output.mp4)'
    )
    
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.25,
        help='Порог уверенности для детекции'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Устройство (auto, cpu, cuda:0, и т.д.)'
    )
    
    parser.add_argument(
        '--small-objects',
        action='store_true',
        help='Улучшенная детекция маленьких объектов (тайлинг)'
    )
    parser.add_argument(
        '--tile-size',
        type=int,
        default=640,
        help='Размер тайла для маленьких объектов'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.3,
        help='Перекрытие тайлов 0.0-1.0'
    )
    
    parser.add_argument(
        '--tracking',
        action='store_true',
        help='Включить трекинг объектов (ByteTrack)'
    )
    parser.add_argument(
        '--track-threshold',
        type=float,
        default=0.5,
        help='Порог уверенности для трекинга'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Не показывать видео во время обработки'
    )
    parser.add_argument(
        '--show-info-panel',
        action='store_true',
        help='Показывать информационную панель на видео'
    )
    
    parser.add_argument(
        '--save-metrics',
        action='store_true',
        help='Сохранить метрики в JSON'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Максимальное количество кадров'
    )
    
    return parser.parse_args()


def main():
    """Главная функция для запуска детекции"""
    args = parse_args()
    
    print("=" * 60)
    print("Детекция людей на видео")
    print("=" * 60)
    print(f"Модель: {args.model.upper()}")
    print(f"Входное видео: {args.input}")
    print(f"Порог уверенности: {args.conf_threshold}")
    print(f"Трекинг: {'Включен' if args.tracking else 'Выключен'}")
    print(f"Детекция маленьких объектов: {'Включена (тайлинг)' if args.small_objects else 'Выключена'}")
    if args.small_objects:
        print(f"  Размер тайла: {args.tile_size}x{args.tile_size}")
        print(f"  Перекрытие: {args.overlap*100:.0f}%")
    print(f"Устройство: {args.device}")
    print("=" * 60)
    

    if args.output is None:
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        args.output = str(output_dir / f"{args.model}_output.mp4")
    
    print(f"\nИнициализация детектора {args.model.upper()}...")
    detector = get_detector(
        args.model,
        device=args.device,
        confidence_threshold=args.conf_threshold,
    )
    
    tracker = None
    if args.tracking:
        print("Инициализация ByteTrack трекера...")
        tracker = ByteTracker(
            track_threshold=args.track_threshold,
            track_buffer=30,
            match_threshold=0.8,
        )
    
    print(f"\nОткрываем видео: {args.input}")
    video_reader = VideoReader(args.input)
    
    print(f"Создаем выходной файл: {args.output}")
    video_writer = VideoWriter(
        args.output,
        fps=video_reader.fps,
        frame_size=(video_reader.width, video_reader.height)
    )
    
   
    metrics = DetectionMetrics()
    metrics.model_name = args.model.upper()
    
    
    fps_counter = FPSCounter(window_size=30)
    
    print("\nОбработка видео...")
    print("-" * 60)
    
    frame_count = 0
    max_frames = args.max_frames if args.max_frames else video_reader.frame_count
    
    try:
        with tqdm(total=min(max_frames, video_reader.frame_count), desc="Обработка") as pbar:
            while True:
                
                ret, frame = video_reader.read()
                if not ret or (args.max_frames and frame_count >= args.max_frames):
                    break
                
                frame_count += 1
                
                
                start_time = time.time()
                if args.small_objects:
                    
                    detections = detect_with_tiling(
                        frame,
                        detector,
                        tile_size=args.tile_size,
                        overlap=args.overlap,
                        conf_threshold=args.conf_threshold,
                        nms_iou=0.5
                    )
                else:
                    
                    detections = detector.detect(frame)
                inference_time = time.time() - start_time
                
                
                if tracker is not None and detections:
                    detections = tracker.update(detections)
                
                
                metrics.add_frame_result(detections, inference_time)
                
                
                current_fps = fps_counter.update()
                
                
                frame_vis = draw_detections(
                    frame,
                    detections,
                    show_confidence=True,
                    show_class=False,
                    show_id=args.tracking,
                )
                
                
                if args.show_info_panel:
                    frame_vis = draw_info_panel(
                        frame_vis,
                        model_name=args.model.upper(),
                        fps=current_fps,
                        detection_count=len(detections),
                        frame_number=frame_count,
                        total_frames=video_reader.frame_count,
                    )
                
                
                video_writer.write(frame_vis)
                
                
                pbar.update(1)
                pbar.set_postfix({
                    'FPS': f'{current_fps:.1f}',
                    'Detections': len(detections),
                })
                
                
                if not args.no_display:
                    import cv2
                    cv2.imshow('Детекция людей', frame_vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nПрервано пользователем")
                        break
    
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
    
    finally:
        
        video_reader.release()
        video_writer.release()
        
        if not args.no_display:
            import cv2
            cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("Обработка завершена!")
    print("=" * 60)
    print(metrics.summary())
    print("=" * 60)
    print(f"Результат сохранен в: {args.output}")
    
    if args.save_metrics:
        import json
        metrics_path = Path(args.output).parent / f"{args.model}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"Метрики сохранены в: {metrics_path}")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
