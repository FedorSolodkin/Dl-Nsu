import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any

def create_model(config: Dict[str, Any]) -> nn.Module:
    model_name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]
    pretrained = config["model"]["pretrained"]
    device = config["train"]["device"]
    
    if model_name=="resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    else:
        raise ValueError (f"Модель {model_name} не поддерживается")
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features,num_classes)
    
    model = model.to(device)
    
    return model

def get_optimizer( model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    
    lr = config["train"]["learning_rate"]
    weight_decay = config["train"]["weight_decay"]
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    return optimizer

def get_criterion(config: Dict[str, Any]) -> nn.Module:
    return nn.CrossEntropyLoss()
