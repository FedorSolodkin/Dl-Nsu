from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from PIL import Image
import torch
from typing import Dict, Any, Tuple

class SimpsonsDataset(Dataset):
    def __init__(self, data_dir:str,transform=None):
        self.data_dir = Path(data_dir)/"simpsons_dataset"
        self.transform = transform
        exclude_names = {'simpsons_dataset', 'test', 'temp'}
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir() and d.name not in exclude_names])
        
        self.class_to_idx = {}
        for idx, cls in enumerate(self.classes):
            self.class_to_idx[cls] = idx  
            
        self.samples =[]
        for cls in self.classes:
            for img_path in  (self.data_dir/cls).glob("*.jpg"):
                self.samples.append((img_path,self.class_to_idx[cls]))
    
    def __len__(self) ->int:
        return len(self.samples)
    
    def __getitem__ (self, idx:int ) -> Tuple[torch.Tensor,int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        return image,label
    
    @property
    def num_classes(self)->int:
        return len(self.classes)
    
def get_transforms(config: Dict[str, Any], is_train: bool = True):
    img_size = config["data"]["img_size"]
    
    normalize = transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std = [0.229,0.224,0.225]
    )
    
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    
    
def get_dataloaders(config:Dict[str,Any])-> Tuple[DataLoader,DataLoader,int]:
    
    train_transform = get_transforms(config,is_train=True)
    val_transform = get_transforms(config,is_train=False)
    full_dataset = SimpsonsDataset(config["data"]["data_dir"],transform=val_transform)
    
    train_size = int(len(full_dataset)*config["data"]["train_split"])
    val_size = len(full_dataset)-train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size,val_size],
        generator = torch.Generator().manual_seed(config["train"]["seed"])
        
    )
    
    train_dataset.dataset.transform = train_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle= True,
        num_workers = config["data"]["num_workers"],
        pin_memory = True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle= False,
        num_workers = config["data"]["num_workers"],
        pin_memory = True
    )
    return train_loader, val_loader, full_dataset.num_classes