"""
Бенчмарк YOLOv8 и RT-DETR.
Замеряет все важные метрики: скорость, точность, количество детекций.
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from detectors import get_detector
from utils import VideoReader
from utils.metrics import DetectionMetrics


def benchmark_model(model_name, video_path, device='auto', max_frames=None, conf_threshold=0.25):
    
    print(f"\n{'='*60}")
    print(f"Бенчмарк {model_name.upper()}")
    print(f"{'='*60}")
    
    # Инициализация детектора
    detector = get_detector(model_name, device=device, confidence_threshold=conf_threshold)
    
    # Открытие видео
    video_reader = VideoReader(video_path)
    

    metrics = DetectionMetrics()
    metrics.model_name = model_name.upper()
    metrics.input_size = (640, 640)
    
    # Размер модели
    metrics.model_size_mb = 0.0
    
    # Обработка кадров
    frame_count = 0
    max_frames = max_frames if max_frames else video_reader.frame_count
    
    print(f"Обработка {min(max_frames, video_reader.frame_count)} кадров...")
    
   
    ret, frame = video_reader.read()
    if ret:
        _ = detector.detect(frame)
    video_reader.seek(0)
    
    
    while True:
        ret, frame = video_reader.read()
        if not ret or frame_count >= max_frames:
            break
        
        frame_count += 1
        
        # Замер времени инференса
        start_time = time.time()
        detections = detector.detect(frame)
        inference_time = time.time() - start_time
        
        # Обновление метрик
        metrics.add_frame_result(detections, inference_time)
        
        if frame_count % 100 == 0:
            print(f"Обработано {frame_count}/{max_frames} кадров | "
                  f"FPS: {metrics.average_fps:.2f} | "
                  f"Детекций: {len(detections)}")
    
    video_reader.release()
    
    print(f"\n{metrics.summary()}")
    
    return metrics


def compare_models(results_dict):
    
    output_dir = Path('report/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Стиль графиков
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
   
    models = list(results_dict.keys())
    
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fps_values = [results_dict[m].average_fps for m in models]
    colors = ['#2ecc71', '#3498db']
    bars = ax.bar(models, fps_values, color=colors, alpha=0.8)
    
    # Добавляем значения на столбцы
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Frames Per Second (FPS)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: FPS', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fps_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Сохранено: {output_dir / 'fps_comparison.png'}")
    plt.close()
    
    # 2. Сравнение времени инференса
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    inf_time_values = [results_dict[m].average_inference_time for m in models]
    bars = ax.bar(models, inf_time_values, color=colors, alpha=0.8)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} ms',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: Inference Time per Frame', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_time_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Сохранено: {output_dir / 'inference_time_comparison.png'}")
    plt.close()
    
    # 3. Статистика детекций
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Среднее количество детекций на кадр
    avg_det_values = [results_dict[m].average_detections_per_frame for m in models]
    axes[0].bar(models, avg_det_values, color=colors, alpha=0.8)
    axes[0].set_ylabel('Average Detections per Frame', fontsize=11, fontweight='bold')
    axes[0].set_title('Detection Count Comparison', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, (model, value) in enumerate(zip(models, avg_det_values)):
        axes[0].text(i, value, f'{value:.2f}', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Средняя уверенность
    avg_conf_values = [results_dict[m].average_confidence for m in models]
    axes[1].bar(models, avg_conf_values, color=colors, alpha=0.8)
    axes[1].set_ylabel('Average Confidence Score', fontsize=11, fontweight='bold')
    axes[1].set_title('Confidence Comparison', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, (model, value) in enumerate(zip(models, avg_conf_values)):
        axes[1].text(i, value, f'{value:.3f}', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detection_stats_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Сохранено: {output_dir / 'detection_stats_comparison.png'}")
    plt.close()
    
    # 4. Таблица сравнения
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    table_data = []
    headers = ['Metric', 'YOLOv8', 'RT-DETR', 'Winner']
    
    # Подготовка данных
    metrics_to_compare = [
        ('FPS (↑)', 'average_fps', True),
        ('Inference Time (ms) (↓)', 'average_inference_time', False),
        ('Avg Detections/Frame', 'average_detections_per_frame', None),
        ('Avg Confidence (↑)', 'average_confidence', True),
        ('Total Detections', 'total_detections', None),
    ]
    
    for metric_name, attr, higher_better in metrics_to_compare:
        yolo_val = getattr(results_dict['yolov8'], attr)
        rtdetr_val = getattr(results_dict['rtdetr'], attr)
        
        # Determine winner
        if higher_better is not None:
            if higher_better:
                winner = 'YOLOv8' if yolo_val > rtdetr_val else 'RT-DETR'
            else:
                winner = 'YOLOv8' if yolo_val < rtdetr_val else 'RT-DETR'
        else:
            winner = '-'
        
        # Format values
        if 'time' in attr.lower() or 'fps' in attr.lower():
            yolo_str = f'{yolo_val:.2f}'
            rtdetr_str = f'{rtdetr_val:.2f}'
        elif 'confidence' in attr.lower():
            yolo_str = f'{yolo_val:.3f}'
            rtdetr_str = f'{rtdetr_val:.3f}'
        else:
            yolo_str = f'{yolo_val:.2f}' if isinstance(yolo_val, float) else str(yolo_val)
            rtdetr_str = f'{rtdetr_val:.2f}' if isinstance(rtdetr_val, float) else str(rtdetr_val)
        
        table_data.append([metric_name, yolo_str, rtdetr_str, winner])
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.3, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Стиль заголовка
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Стиль ячеек
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if j == 3 and table_data[i-1][3] != '-':
                table[(i, j)].set_facecolor('#2ecc71' if table_data[i-1][3] == 'YOLOv8' else '#e74c3c')
                table[(i, j)].set_text_props(weight='bold', color='white')
    
    plt.title('Model Comparison Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_table.png', dpi=300, bbox_inches='tight')
    print(f"Сохранено: {output_dir / 'comparison_table.png'}")
    plt.close()
    
    print(f"\nВсе графики сохранены в: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Бенчмарк моделей детекции')
    parser.add_argument('--input', type=str, default='crowd.mp4', help='Входное видео')
    parser.add_argument('--output', type=str, default='results/benchmark_results.json',
                       help='Выходной JSON файл')
    parser.add_argument('--device', type=str, default='auto', help='Устройство (auto, cpu, cuda)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Максимум кадров для обработки (по умолчанию: все)')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='Порог уверенности')
    
    args = parser.parse_args()
    
    # Создание выходной директории
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Бенчмарк обеих моделей
    results = {}
    
    for model_name in ['yolov8', 'rtdetr']:
        metrics = benchmark_model(
            model_name,
            args.input,
            device=args.device,
            max_frames=args.max_frames,
            conf_threshold=args.conf_threshold
        )
        results[model_name] = metrics
    
    # Создание графиков сравнения
    print(f"\n{'='*60}")
    print("Создание графиков сравнения...")
    print(f"{'='*60}")
    compare_models(results)
    
    # Сохранение результатов в JSON
    results_dict = {
        model_name: metrics.to_dict()
        for model_name, metrics in results.items()
    }
    
    with open(args.output, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Результаты бенчмарка сохранены в: {args.output}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
