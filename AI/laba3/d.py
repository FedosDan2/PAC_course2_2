import os
import shutil
import random
import pandas as pd
import ast
from PIL import Image
from collections import defaultdict

# ====================== ПУТИ ======================
DATASET_PATH = '/home/f.danilov/.cache/kagglehub/datasets/airbusgeo/airbus-aircrafts-sample-dataset/versions/3'
IMAGES_DIR = os.path.join(DATASET_PATH, 'images')
CSV_FILE = os.path.join(DATASET_PATH, 'annotations.csv')
OUTPUT_DIR = '/home/f.danilov/workflow/PAC_course2_2/AI/laba3/dataset_yolo'

# ====================== НАСТРОЙКА КЛАССОВ ======================
# Оставляем ТОЛЬКО этот класс
TARGET_CLASS = 'Airplane'

# ====================== ЧТЕНИЕ CSV ======================
print("📖 Чтение аннотаций из CSV...")
df = pd.read_csv(CSV_FILE)
print(f"  Всего записей: {len(df)}")
print(f"  Уникальных классов: {df['class'].unique()}")

# Фильтруем - оставляем только Airplane
print(f"\n🔍 Фильтрация: оставляем только класс '{TARGET_CLASS}'...")
df_filtered = df[df['class'] == TARGET_CLASS].copy()
print(f"  Записей после фильтрации: {len(df_filtered)}")
print(f"  Уникальных изображений: {df_filtered['image_id'].nunique()}")

# Проверяем, какие классы были исключены
excluded_classes = set(df['class'].unique()) - {TARGET_CLASS}
if excluded_classes:
    excluded_count = len(df[df['class'].isin(excluded_classes)])
    print(f"  Исключено классов: {excluded_classes}")
    print(f"  Исключено записей: {excluded_count}")

# ====================== КОНВЕРТАЦИЯ АННОТАЦИЙ ======================
print("\n🔄 Конвертация аннотаций в формат YOLO...")

# Только один класс с индексом 0
class_to_idx = {TARGET_CLASS: 0}
print(f"  Класс: {TARGET_CLASS} -> индекс 0")

# Группируем аннотации по изображениям
annotations_by_image = defaultdict(list)

for _, row in df_filtered.iterrows():
    image_id = row['image_id']
    
    # Парсим геометрию
    try:
        geometry = ast.literal_eval(row['geometry'])
    except:
        print(f"  ⚠️ Ошибка парсинга геометрии для {image_id}")
        continue
    
    annotations_by_image[image_id].append({
        'class_idx': 0,  # всегда 0, так как только один класс
        'polygon': geometry
    })

print(f"  Изображений с аннотациями: {len(annotations_by_image)}")

# ====================== СОЗДАНИЕ ФАЙЛОВ РАЗМЕТКИ YOLO ======================
print("\n📝 Создание временных файлов разметки...")

TEMP_LABELS_DIR = '/tmp/yolo_labels_temp'
os.makedirs(TEMP_LABELS_DIR, exist_ok=True)

valid_images = []
skipped_no_annotation = 0
skipped_bad_image = 0

for image_file in sorted(os.listdir(IMAGES_DIR)):
    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    image_path = os.path.join(IMAGES_DIR, image_file)
    
    # Пропускаем изображения без аннотаций (если нужно)
    if image_file not in annotations_by_image:
        skipped_no_annotation += 1
        continue  # Закомментируйте эту строку, если хотите включить изображения без самолётов
    
    # Получаем размеры изображения
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"  ⚠️ Не удалось прочитать {image_file}: {e}")
        skipped_bad_image += 1
        continue
    
    # Создаём файл разметки
    base_name = os.path.splitext(image_file)[0]
    label_path = os.path.join(TEMP_LABELS_DIR, base_name + '.txt')
    
    with open(label_path, 'w') as f:
        for ann in annotations_by_image[image_file]:
            polygon = ann['polygon']
            
            # Нормализуем координаты
            normalized_coords = []
            for x, y in polygon:
                x_norm = x / img_width
                y_norm = y / img_height
                normalized_coords.extend([x_norm, y_norm])
            
            # Формат строки: 0 x1 y1 x2 y2 ... xn yn (0 - всегда класс airplane)
            coords_str = ' '.join([f'{coord:.6f}' for coord in normalized_coords])
            f.write(f"0 {coords_str}\n")
    
    valid_images.append({
        'image_path': image_path,
        'label_path': label_path,
        'image_id': base_name
    })

print(f"  ✓ Создано файлов разметки: {len(valid_images)}")
print(f"  ⚠️ Пропущено (без аннотаций): {skipped_no_annotation}")
print(f"  ⚠️ Пропущено (битые): {skipped_bad_image}")

# ====================== РАЗБИЕНИЕ НА ВЫБОРКИ ======================
print("\n✂️ Разбиение на train/val/test...")

image_ids = [item['image_id'] for item in valid_images]
data_len = len(image_ids)

# Если данных мало, адаптируем разбиение
if data_len < 10:
    # Очень мало данных - только train и val
    train_count = max(1, int(0.8 * data_len))
    valid_count = data_len - train_count
    test_count = 0
    print("  ⚠️ Мало данных! Создаём только train и val")
else:
    test_count = max(1, int(0.2 * data_len))
    trainval_count = data_len - test_count
    valid_count = max(1, int(0.2 * trainval_count))
    train_count = trainval_count - valid_count

print(f"  Train: {train_count}, Valid: {valid_count}, Test: {test_count}")

random.seed(42)
random.shuffle(image_ids)

train_ids = set(image_ids[:train_count])
valid_ids = set(image_ids[train_count:train_count + valid_count])
test_ids = set(image_ids[train_count + valid_count:])

# ====================== СОЗДАНИЕ СТРУКТУРЫ YOLO ======================
print("\n📁 Создание структуры датасета YOLO...")

# Удаляем старую папку если есть
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# Создаём директории
splits_to_create = ['train', 'val']
if test_count > 0:
    splits_to_create.append('test')

for split in splits_to_create:
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)

# Копируем файлы
copied = {split: 0 for split in splits_to_create}

for item in valid_images:
    image_id = item['image_id']
    src_img = item['image_path']
    src_lbl = item['label_path']
    
    # Определяем сплит
    if image_id in train_ids:
        split = 'train'
    elif image_id in valid_ids:
        split = 'val'
    else:
        split = 'test'
    
    # Копируем изображение
    img_ext = os.path.splitext(src_img)[1]
    dst_img = os.path.join(OUTPUT_DIR, split, 'images', f"{image_id}{img_ext}")
    shutil.copy2(src_img, dst_img)
    
    # Копируем разметку
    dst_lbl = os.path.join(OUTPUT_DIR, split, 'labels', f"{image_id}.txt")
    shutil.copy2(src_lbl, dst_lbl)
    
    copied[split] += 1

for split in splits_to_create:
    print(f"  ✓ {split}: {copied[split]} файлов")

# ====================== СОЗДАНИЕ data.yaml ======================
print("\n📄 Создание data.yaml...")

yaml_content = f"""# YOLO Dataset Configuration
# Dataset: Airbus Aircrafts - Airplane Only
path: {OUTPUT_DIR}
train: train/images
val: val/images
test: {'test/images' if test_count > 0 else ''}

# Single class - Airplane
nc: 1
names:
  0: airplane
"""

yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"  ✓ Сохранён в {yaml_path}")
print(f"  Содержимое data.yaml:")
print(f"    path: {OUTPUT_DIR}")
print(f"    nc: 1")
print(f"    names: {{0: airplane}}")

# ====================== ОЧИСТКА ВРЕМЕННЫХ ФАЙЛОВ ======================
shutil.rmtree(TEMP_LABELS_DIR)

# ====================== СТАТИСТИКА ======================
print("\n" + "="*70)
print("📊 СТАТИСТИКА ДАТАСЕТА")
print("="*70)
print(f"  Класс: airplane (индекс 0)")
print(f"  Всего изображений: {len(valid_images)}")
print(f"  ├── Train: {copied.get('train', 0)}")
print(f"  ├── Val: {copied.get('val', 0)}")
print(f"  └── Test: {copied.get('test', 0)}")

# Подсчитываем общее количество самолётов
total_planes = sum(len(annotations_by_image.get(item['image_id']+'.jpg', [])) 
                   for item in valid_images)
print(f"  Всего самолётов: {total_planes}")
print(f"  В среднем на изображение: {total_planes/len(valid_images):.1f}")

print(f"\n{'='*70}")
print("✅ ГОТОВО!")
print(f"{'='*70}")
print(f"\nДатасет создан в: {OUTPUT_DIR}")
print(f"\nКоманда для обучения сегментации:")
print(f"yolo segment train data={yaml_path} model=yolo11n-seg.pt epochs=100 imgsz=640")
print(f"{'='*70}")

# ====================== ПРОВЕРКА РАЗМЕТКИ ======================
print("\n🔍 Проверка первых 3 файлов разметки...")
label_dir = os.path.join(OUTPUT_DIR, 'train', 'labels')
label_files = sorted(os.listdir(label_dir))[:3]

for lf in label_files:
    lf_path = os.path.join(label_dir, lf)
    with open(lf_path, 'r') as f:
        lines = f.readlines()
    print(f"\n  {lf}:")
    for i, line in enumerate(lines[:2]):  # показываем первые 2 строки
        parts = line.strip().split()
        class_idx = parts[0]
        coords_count = len(parts[1:])
        print(f"    Объект {i+1}: class={class_idx}, точек={coords_count//2}")
    if len(lines) > 2:
        print(f"    ... и ещё {len(lines)-2} объектов")