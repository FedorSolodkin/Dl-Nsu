import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Tuple


def validate(
    model: Module,
    dataloader: DataLoader,
    criterion: Module,
    device: str
) -> Tuple[float, float]:
    model.eval() 
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad(): 
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy        