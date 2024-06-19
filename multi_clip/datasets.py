from typing import List

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from .processors.blip_processor import BlipProcessor
from .processors.clip_processor import ClipProcessor

class ClipDataset(Dataset):
    def __init__(self, 
                 image_file_paths: List[str], 
                 texts: List[str], 
                 labels: List[List[int]],
                 pretrained_model_name_or_path="openai/clip-vit-large-patch14"):
        
        self.image_file_paths = image_file_paths
        self.texts = texts
        self.labels = labels
        
        self.processor = ClipProcessor(pretrained_model_name_or_path)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_file_paths[idx])
        text = self.texts[idx]

        image_data, text_data = self.processor(image, text)

        pixel_values = image_data["pixel_values"].squeeze(0)
        input_ids = text_data["input_ids"].squeeze(0)
        attention_mask = text_data["attention_mask"].squeeze(0)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return (pixel_values, input_ids, attention_mask), label

class BlipDataset(Dataset):
    def __init__(self, 
                 image_paths: List[str], 
                 texts: List[str], 
                 labels: List[List[int]], 
                 pretrained_model_name_or_path="Salesforce/blip-itm-large-coco"):
        
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels

        self.processor = BlipProcessor(pretrained_model_name_or_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with Image.open(image_path) as image:
            raw_image = image.convert("RGB")

        inputs = self.processor(image=raw_image, text=self.texts[idx])

        pixel_values = inputs["pixel_values"].squeeze(0)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        label = self.labels[idx]
        
        if isinstance(label, np.ndarray):
            label = label.tolist()
        
        label = torch.tensor(label, dtype=torch.float32)
        
        return (pixel_values, input_ids, attention_mask), label

class BlipEmbedDataset(Dataset):
    def __init__(self, 
                 embeddings: np.ndarray, 
                 labels: List[List[int]]):
        
        self.embeddings = torch.from_numpy(embeddings)
        self.labels = labels
    
    def __len__(self):
        return self.embeddings.size(0)
    
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return embedding, label