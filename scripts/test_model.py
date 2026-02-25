import yaml
import torch
from src.model import create_model, get_optimizer, get_criterion

def test_model():
    with open("config/train_config.yaml","r") as f:
        config = yaml.safe_load(f)
    
    print("создаем модель")
    model = create_model(config)
    
    device = config["train"]["device"]
    print(f"Девайся:{device}")
    print(f"Модель на устройстве: {next(model.parameters()).device}")
    dummy_input = torch.randn(32, 3, 224, 224).to(device)
    with torch.no_grad(): 
        output = model(dummy_input)
    
    print(f"Shape входа: {dummy_input.shape}")
    print(f"Shape выхода: {output.shape}")
    print(f"Ожидаемый выход: [32, 42] (batch_size, num_classes)")
    
    optimizer = get_optimizer(model, config)
    criterion = get_criterion(config)
    
    print(f"Оптимизатор: {optimizer.__class__.__name__}")
    print(f"Функция потерь: {criterion.__class__.__name__}")
    
    fake_labels = torch.randint(0, 42, (32,)).to(device)
    loss = criterion(output, fake_labels)
    print(f"Тестовый Loss: {loss.item():.4f}")
    
if __name__ == "__main__":
    test_model()