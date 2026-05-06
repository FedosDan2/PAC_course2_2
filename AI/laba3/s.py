import os
import numpy as np
from PIL import Image

def polygon_to_bbox(polygon_coords):
    """Конвертирует полигон в bounding box [x_center, y_center, width, height]"""
    # Извлекаем x и y координаты
    xs = polygon_coords[0::2]
    ys = polygon_coords[1::2]
    
    # Находим границы
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    
    # Конвертируем в формат YOLO: [x_center, y_center, width, height]
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height

# Конвертируем все файлы разметки
DATASET_DIR = '/home/f.danilov/workflow/PAC_course2_2/AI/laba3/dataset'

for split in ['train', 'val', 'test']:
    labels_dir = os.path.join(DATASET_DIR, split, 'labels')
    
    if not os.path.exists(labels_dir):
        continue
    
    print(f"\n📁 Конвертация {split}...")
    converted = 0
    
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        
        label_path = os.path.join(labels_dir, label_file)
        
        # Читаем старую разметку
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Конвертируем и записываем новую
        with open(label_path, 'w') as f:
            for line in lines:
                parts = line.strip().split()
                class_idx = parts[0]
                coords = [float(x) for x in parts[1:]]
                
                # Конвертируем полигон в bbox
                x_c, y_c, w, h = polygon_to_bbox(coords)
                
                # Формат детекции YOLO: class x_center y_center width height
                f.write(f"{class_idx} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                converted += 1
    
    print(f"  ✓ Конвертировано объектов: {converted}")

print("\n✅ Готово! Разметка переведена в формат детекции (bounding boxes)")