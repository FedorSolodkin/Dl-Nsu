# scripts/train.py
import yaml
import torch
import json
from pathlib import Path
from src.utils import set_seed
from src.model import create_model, get_optimizer, get_criterion
from src.dataset import get_dataloaders
from src.train import train_one_epoch
from src.validate import validate

def main():
    
    with open("config/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
 
    set_seed(config["train"]["seed"])
    
 
    artifacts_dir = Path(config["artifacts"]["save_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    
    print("Загружаем датасет...")
    train_loader, val_loader, num_classes = get_dataloaders(config)
    print(f"Train: {len(train_loader)} батчей, Val: {len(val_loader)} батчей")
    

    print("Создаём модель...")
    device = config["train"]["device"]
    

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA недоступна, переключаемся на CPU")
        device = "cpu"
        config["train"]["device"] = device
    
    model = create_model(config)
    optimizer = get_optimizer(model, config)
    criterion = get_criterion(config)
    
    print(f"Устройство: {device}")
    print(f"Классов: {num_classes}")
    

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    

    print("Начинаем обучение...")
    for epoch in range(1, config["train"]["epochs"] + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
       
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
     
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        
        print(f"\nEpoch {epoch}/{config['train']['epochs']}")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                artifacts_dir / "best_model.pth"
            )
            print(f" Сохранена лучшая модель (Val Acc: {val_acc:.2f}%)")
    
 
    with open(artifacts_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
   
    with open(artifacts_dir / "config_used.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nОбучение завершено!")
    print(f"Лучшая валидационная точность: {best_val_acc:.2f}%")
    print(f"Модель сохранена в: {artifacts_dir / 'best_model.pth'}")

if __name__ == "__main__":
    main()