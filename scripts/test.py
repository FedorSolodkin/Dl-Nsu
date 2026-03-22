# scripts/test_kaggle.py
import yaml
import torch
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from src.utils import set_seed
from src.model import create_model, get_criterion
from src.dataset import KaggleTestDataset, get_transforms
from torch.utils.data import DataLoader

def test_kaggle():
    # 1. Загружаем конфиг
    with open("config/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print(f"data_dir из конфига: {config['data']['test_dir']}")
    test_path = Path(config["data"]["test_dir"]) 
    print(f"Путь к тесту: {test_path}")
    print(f"Папка существует: {test_path.exists()}")

    if test_path.exists():
        files = list(test_path.glob("*.jpg"))
        print(f"Найдено файлов: {len(files)}")
    else:
        print("Папка не найдена!")
    
    
    
    set_seed(config["train"]["seed"])
    
    # 2. Проверяем наличие весов модели
    artifacts_dir = Path(config["artifacts"]["save_dir"])
    model_path = artifacts_dir / "best_model.pth"
    
    if not model_path.exists():
        print(f"Файл модели не найден: {model_path}")
        print("Сначала запустите обучение: python scripts/train.py")
        return
    
    # 3. Загружаем тестовый датасет Kaggle
    print("📥 Загружаем Kaggle test set...")
    
    test_transform = get_transforms(config, is_train=False)
    

    test_dataset = KaggleTestDataset(
        config["data"]["data_dir"],config["data"]["test_dir"],
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )
    device = config["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model= create_model(config)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    criterion = get_criterion(config)
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    
    print(f"\n📊 Результаты тестирования (Kaggle test set):")
    print(f"   Test Loss: {avg_loss:.4f}")
    print(f"   Test Accuracy: {accuracy:.2f}%")
    print(f"   Правильно: {correct}/{total}")
    class_names = test_dataset.classes
    predictions_detail = []
    for idx, (img_path, label) in enumerate(test_dataset.samples):
        predictions_detail.append({
            "filename": img_path.name,
            "true_class": class_names[all_labels[idx]],
            "predicted_class": class_names[all_preds[idx]],
            "confidence": float(max(all_probs[idx])),
            "correct": bool(all_preds[idx] == all_labels[idx])
        })

    with open(artifacts_dir / "kaggle_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions_detail, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    test_kaggle()