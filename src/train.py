import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
from tqdm import tqdm

def train_one_epoch(
    model: Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Module,
    device: str,   
    epoch: int
) -> Tuple[float,float]:
    
    model.train()
    
    total_loss = 0
    correct =0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for images,labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs,labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy