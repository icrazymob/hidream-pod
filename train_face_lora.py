#!/usr/bin/env python3
"""
Скрипт для тренировки LoRA лица с HiDream I1
Основан на лучших практиках сообщества и оптимизирован для 10-20 фотографий лица
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
import time

def setup_environment():
    """Настройка окружения и проверка зависимостей"""
    print("🔧 Настройка окружения...")
    
    # Проверка CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA не доступна!")
        print(f"✅ CUDA доступна: {torch.cuda.get_device_name(0)}")
        print(f"📊 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("❌ PyTorch не установлен!")
        sys.exit(1)
    
    # Проверка HuggingFace токена
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("⚠️  HF_TOKEN не установлен. Убедитесь, что у вас есть доступ к gated моделям.")

def prepare_dataset(input_dir, output_dir, trigger_word):
    """Подготовка датасета для тренировки"""
    print(f"📂 Подготовка датасета из {input_dir}...")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ Директория {input_dir} не существует!")
        sys.exit(1)
    
    # Создаем выходную директорию
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Поддерживаемые форматы изображений
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    # Копируем и переименовываем изображения
    image_count = 0
    for img_file in input_path.iterdir():
        if img_file.suffix.lower() in image_extensions:
            # Новое имя файла
            new_name = f"{trigger_word}_{image_count:03d}{img_file.suffix}"
            new_path = output_path / new_name
            
            # Копируем файл
            shutil.copy2(img_file, new_path)
            
            # Создаем текстовое описание
            caption_path = new_path.with_suffix('.txt')
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(f"фото {trigger_word}, высокое качество, детализированное лицо")
            
            image_count += 1
            print(f"✅ Обработано: {new_name}")
    
    if image_count == 0:
        print("❌ Не найдено изображений для обработки!")
        sys.exit(1)
    
    print(f"📊 Обработано {image_count} изображений")
    
    # Рекомендации по количеству шагов
    if image_count < 10:
        recommended_steps = 800
        print("⚠️  Мало изображений (<10). Рекомендуется добавить больше фото.")
    elif image_count <= 20:
        recommended_steps = 1200
    else:
        recommended_steps = 1500
        print("⚠️  Много изображений (>20). Возможно переобучение.")
    
    print(f"💡 Рекомендуемое количество шагов: {recommended_steps}")
    return image_count, recommended_steps

def update_config(config_path, trigger_word, dataset_path, steps):
    """Обновление конфигурации с пользовательскими параметрами"""
    print("⚙️  Обновление конфигурации...")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = f.read()
    
    # Обновляем путь к датасету
    config = config.replace('/workspace/training_data', str(dataset_path))
    
    # Обновляем количество шагов
    config = config.replace('steps: 1500', f'steps: {steps}')
    
    # Обновляем промпты с trigger word
    config = config.replace('[trigger]', trigger_word)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config)
    
    print("✅ Конфигурация обновлена")

def run_training(config_path):
    """Запуск тренировки"""
    print("🚀 Запуск тренировки...")
    print("📝 Логи будут сохранены в training.log")
    
    cmd = [
        'python', '/workspace/ai-toolkit/run.py',
        config_path
    ]
    
    # Запуск с логированием
    with open('/workspace/training.log', 'w') as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line.strip())
            log_file.write(line)
            log_file.flush()
        
        process.wait()
        return process.returncode

def main():
    parser = argparse.ArgumentParser(description='Тренировка LoRA лица для HiDream I1')
    parser.add_argument('--input_dir', required=True, help='Директория с фотографиями')
    parser.add_argument('--trigger_word', required=True, help='Триггер-слово для активации LoRA')
    parser.add_argument('--output_dir', default='/workspace/output', help='Директория для сохранения модели')
    parser.add_argument('--steps', type=int, help='Количество шагов тренировки (авто если не указано)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎭 HiDream I1 Face LoRA Trainer")
    print("=" * 60)
    
    # Настройка окружения
    setup_environment()
    
    # Подготовка датасета
    dataset_dir = '/workspace/training_data'
    image_count, recommended_steps = prepare_dataset(
        args.input_dir, 
        dataset_dir, 
        args.trigger_word
    )
    
    # Определяем количество шагов (адаптировано под проверенные настройки RTX 3090)
    if args.steps:
        steps = args.steps
    else:
        # Основано на успешном опыте с diffusion-pipe: 111 фото = ~4440 шагов (10 эпох)
        # Адаптируем для меньшего количества фото
        if image_count <= 10:
            steps = 600
        elif image_count <= 15:
            steps = 800  
        elif image_count <= 20:
            steps = 1000
        else:
            steps = 1200
            print("⚠️  Много изображений (>20). Возможно переобучение.")
    
    # Обновляем конфигурацию
    config_path = '/workspace/train_config.yaml'
    update_config(config_path, args.trigger_word, dataset_dir, steps)
    
    # Создаем выходную директорию
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Триггер-слово: {args.trigger_word}")
    print(f"📊 Изображений: {image_count}")
    print(f"🔢 Шагов тренировки: {steps}")
    print("")
    
    # Запуск тренировки
    start_time = time.time()
    return_code = run_training(config_path)
    
    training_time = time.time() - start_time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    
    if return_code == 0:
        print("=" * 60)
        print("🎉 Тренировка завершена успешно!")
        print(f"⏱️  Время тренировки: {hours}ч {minutes}м")
        print(f"📁 Модель сохранена в: {args.output_dir}")
        print("=" * 60)
        
        # Показываем созданные файлы
        output_path = Path(args.output_dir)
        lora_files = list(output_path.glob("*.safetensors"))
        if lora_files:
            print("📦 Созданные LoRA файлы:")
            for lora_file in lora_files:
                print(f"   • {lora_file.name}")
    else:
        print("❌ Тренировка завершилась с ошибкой!")
        sys.exit(return_code)

if __name__ == "__main__":
    main()