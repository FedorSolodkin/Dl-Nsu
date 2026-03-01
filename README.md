# 🍩 Simpsons Character Classification

Классификация персонажей мультсериала "Симпсоны" с использованием глубокого обучения (PyTorch).

## 📋 Описание

Проект решает задачу многоклассовой классификации изображений персонажей из датасета [The Simpsons Characters](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset).

**Задача:** По изображению определить, какой персонаж на нём изображён.

**Модель:** ResNet18 с предобученными весами (Transfer Learning)

**Количество классов:** 42 персонажа



## 🚀 Быстрый старт


```bash
git clone <https://github.com/FedorSolodkin/Dl-Nsu/Lab1>
cd simpson_classification

# Создаём виртуальное окружение
uv venv

# Активируем окружение (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Устанавливаем зависимости
uv sync

# Устанавливаем проект в режиме разработки
uv pip install -e .

python scripts/download_data.py

python scripts/check_data.py

Убедитесь, что найдено 42 класса (персонажей).

Откройте configs/train_config.yaml и укажите путь к датасету:

data:
  data_dir: "C:/Users/<YourUser>/.cache/kagglehub/datasets/.../simpsons_dataset"
  img_size: 224
  batch_size: 32
  num_workers: 0  # Для Windows рекомендуется 0
  train_split: 0.8

model:
  name: "resnet18"
  pretrained: true
  num_classes: 42

train:
  epochs: 10
  learning_rate: 0.001
  weight_decay: 0.0001
  device: "cpu"  # Или "cuda" если есть GPU NVIDIA
  seed: 42

artifacts:
  save_dir: "./artifacts"
  project_name: "simpsons-classification"

🏃 Запуск обучения

python scripts/train.py
```
📊 Что происходит:
Шаг
Описание
1 Загружается датасет (80% train, 20% validation)
2 Создаётся модель ResNet18 с предобученными весами
3 Обучение в течение указанного количества эпох
4 После каждой эпохи — валидация
5 Сохраняется лучшая модель по Validation Accuracy




2. Построение графиков
```bash
python scripts/plot_metrics.py
```

Выход: artifacts/metrics.png — графики Loss и Accuracy по эпохам.

3. Финальный скор
Откройте artifacts/training_history.json и посмотрите последнее значение val_acc:
{
  "val_acc": [35.0, 45.0, 52.0, ..., 72.5]
}


