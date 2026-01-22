"""
Финальный запуск - оба алгоритма со всеми улучшениями
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Запуск команды с отображением прогресса"""
    print("\n" + "=" * 70)
    print(f"Запуск: {description}")
    print("=" * 70)
    
    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] {description} - завершено успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ОШИБКА] {description} - что-то пошло не так")
        return False


def main():
    print("=" * 70)
    print("ДЕТЕКЦИЯ ЛЮДЕЙ")
    print("=" * 70)
    
    # Выбор модели
    print("\nКакие модели запустить?")
    print("  1. Только YOLOv8")
    print("  2. Только RT-DETR")
    print("  3. Обе модели + бенчмарк (рекомендуется)")
    
    model_choice = input("\nВыбор (1/2/3) [по умолчанию: 3]: ").strip()
    if not model_choice:
        model_choice = "3"
    
    run_yolo = model_choice in ["1", "3"]
    run_rtdetr = model_choice in ["2", "3"]
    run_benchmark = model_choice == "3"
    
    # Параметры запуска
    input_video = "crowd.mp4"
    max_frames = input("\nСколько кадров обработать? (Enter = 300): ").strip()
    max_frames = int(max_frames) if max_frames else 300
    
    print(f"\nБудет обработано: {max_frames} кадров")
    print(f"Входное видео: {input_video}")
    if run_yolo:
        print("  - YOLOv8")
    if run_rtdetr:
        print("  - RT-DETR")
    if run_benchmark:
        print("  - Бенчмарк")
    
    input("\nНажмите Enter для старта...\n")
    
    results = {}
    
    # Эксперимент 1: YOLOv8
    if run_yolo:
        cmd_yolo = [
            "python", "inference.py",
            "--model", "yolov8",
            "--input", input_video,
            "--output", "results/yolov8_final.mp4",
            "--tracking",
            "--small-objects",
            "--conf-threshold", "0.30",
            "--save-metrics",
            "--no-display",
            "--max-frames", str(max_frames)
        ]
        results['yolov8'] = run_command(cmd_yolo, "YOLOv8")
    
    # Эксперимент 2: RT-DETR 
    if run_rtdetr:
        cmd_rtdetr = [
            "python", "inference.py",
            "--model", "rtdetr",
            "--input", input_video,
            "--output", "results/rtdetr_final.mp4",
            "--tracking",
            "--small-objects",
            "--conf-threshold", "0.30",
            "--save-metrics",
            "--no-display",
            "--max-frames", str(max_frames)
        ]
        results['rtdetr'] = run_command(cmd_rtdetr, "RT-DETR ")
    
    # Эксперимент 3: Сравнительный бенчмарк
    if run_benchmark:
        cmd_benchmark = [
            "python", "benchmark.py",
            "--input", input_video,
            "--output", "results/benchmark_final.json",
            "--max-frames", str(max_frames)
        ]
        results['benchmark'] = run_command(cmd_benchmark, "Бенчмарк моделей")
    
    # Итоги
    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 70)
    
    success = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nУспешно выполнено: {success} из {total}\n")
    
    for name, ok in results.items():
        status = "[OK]" if ok else "[ОШИБКА]"
        print(f"  {status} {name.upper()}")
    
    # Проверяем созданные файлы
    print("\n" + "=" * 70)
    print("СОЗДАННЫЕ ФАЙЛЫ")
    print("=" * 70)
    
    files = []
    if run_yolo:
        files.extend([
            ("results/yolov8_final.mp4", "Видео YOLOv8"),
            ("results/yolov8_metrics.json", "Метрики YOLOv8"),
        ])
    if run_rtdetr:
        files.extend([
            ("results/rtdetr_final.mp4", "Видео RT-DETR"),
            ("results/rtdetr_metrics.json", "Метрики RT-DETR"),
        ])
    if run_benchmark:
        files.append(("results/benchmark_final.json", "Результаты бенчмарка"))
    
    print()
    for fpath, desc in files:
        path = Path(fpath)
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {desc}: {size:.1f} МБ")
        else:
            print(f"  [НЕ НАЙДЕН] {desc}")
    
    print("\n" + "=" * 70)
    print("ГОТОВО! Все результаты сохранены в папке results/")
    print("=" * 70)
    
    print("\nЧто дальше:")
    step = 1
    if run_yolo or run_rtdetr:
        videos = []
        if run_yolo:
            videos.append("yolov8_final.mp4")
        if run_rtdetr:
            videos.append("rtdetr_final.mp4")
        print(f"  {step}. Просмотрите видео: {' и '.join(videos)}")
        step += 1
    if run_benchmark:
        print(f"  {step}. Изучите метрики: benchmark_final.json")
        step += 1
    print(f"  {step}. Готово")


if __name__ == "__main__":
    main()
