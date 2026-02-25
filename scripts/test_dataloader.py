# scripts/test_dataloader.py
import yaml
from src.dataset import get_dataloaders
from src.utils import set_seed

def test_dataloader():
    # Загружаем конфиг
    with open("config/train_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Фиксируем seed
    set_seed(config["train"]["seed"])
    
    # Получаем DataLoader
    print("📥 Загружаем датасет...")
    train_loader, val_loader, num_classes = get_dataloaders(config)
    
    print(f"✅ Классов: {num_classes}")
    print(f"✅ Train батчей: {len(train_loader)}")
    print(f"✅ Val батчей: {len(val_loader)}")
    
    # Проверяем один батч
    images, labels = next(iter(train_loader))
    print(f"✅ Shape батча: {images.shape}")
    print(f"✅ Метки (первые 10): {labels[:10]}")
    
    print("\n🎉 DataLoader готов!")

if __name__ == "__main__":
    test_dataloader()