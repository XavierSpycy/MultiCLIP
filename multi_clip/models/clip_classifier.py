from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from ..models.router import Router
from ..models.config import CACHE_DIR, CLIP_PRETRAINED_MODEL_NAME_DEFAULT

class ClipClassificationHead(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 num_classes: int,) -> None:
        super(ClipClassificationHead, self).__init__()
        self.cls_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_classes),
        )
    
    def forward(self, features):
        return self.cls_head(features)

class ClipDiscriminator(nn.Module):
    def __init__(self, projection_dim: int, num_classes: int) -> None:
        super(ClipDiscriminator, self).__init__()
        self.cls_head_vision = ClipClassificationHead(projection_dim, num_classes)
        self.cls_head_text = ClipClassificationHead(projection_dim, num_classes)
        self.router = Router(projection_dim, 2)
    
    def get_logits_from_features(self, multimodal_features):
        image_features, text_features = multimodal_features
        logits = self(image_features, text_features)
        return logits
    
    def forward(self, image_features, text_features):
        # Get logits from image and text features
        image_logits = self.cls_head_vision(image_features)
        text_logits = self.cls_head_text(text_features)
        
        # Get weights for image and text features
        image_weights_naive = self.router(image_features)
        text_weights_naive = self.router(text_features)
        weights = F.softmax(torch.cat([image_weights_naive, text_weights_naive], dim=1), dim=1)
        image_weights, text_weights = weights[:, 0], weights[:, 1]
        
        # Combine logits
        logits = image_logits * image_weights[:, None] + text_logits * text_weights[:, None]
        return logits
    
class ClipClassifier(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 pretrained_model_name_or_path: 
                 Optional[str] = None) -> None:
        
        super(ClipClassifier, self).__init__()
        pretrained_model_name_or_path = (
            pretrained_model_name_or_path if pretrained_model_name_or_path is not None else CLIP_PRETRAINED_MODEL_NAME_DEFAULT
        )
        self.feature_extractor: nn.Module = CLIPModel.from_pretrained(pretrained_model_name_or_path, cache_dir=CACHE_DIR)
        self.projection_dim = self.feature_extractor.config.projection_dim
        self.discriminator = ClipDiscriminator(self.projection_dim, num_classes)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def get_image_features(self, pixel_values):
        image_features = self.feature_extractor.get_image_features(pixel_values=pixel_values)
        return F.normalize(image_features, dim=-1)
    
    def get_image_logits_from_features(self, image_features):
        return self.discriminator.cls_head_vision(image_features)
    
    def get_image_logits_from_inputs(self, pixel_values):
        image_features = self.get_image_features(pixel_values)
        return self.get_image_logits_from_features(image_features)
    
    def get_text_features(self, input_ids, attention_mask):
        text_features = self.feature_extractor.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(text_features, dim=-1)
    
    def get_text_logits_from_features(self, text_features):
        return self.discriminator.cls_head_text(text_features)
    
    def get_text_logits_from_inputs(self, input_ids, attention_mask):
        text_features = self.get_text_features(input_ids, attention_mask)
        return self.get_text_logits_from_features(text_features)
    
    def get_multimodal_features(self, input_ids, pixel_values, attention_mask):
        image_features = self.get_image_features(pixel_values)
        text_features = self.get_text_features(input_ids, attention_mask)
        return image_features, text_features
    
    def get_logits_from_features(self, multimodal_features):
        image_features, text_features = multimodal_features
        return self.discriminator(image_features, text_features)
    
    def forward(self, pixel_values, input_ids, attention_mask):
        # Get image and text features
        image_features = self.get_image_features(pixel_values)
        text_features = self.get_text_features(input_ids, attention_mask)
        
        # Get final logits
        logits = self.discriminator(image_features, text_features)
        return logits

class ClipBoostClassifier(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 pretrained_model_name_or_path: 
                 Optional[str] = None) -> None:
        
        super(ClipBoostClassifier, self).__init__()
        pretrained_model_name_or_path = (
            pretrained_model_name_or_path if pretrained_model_name_or_path is not None else CLIP_PRETRAINED_MODEL_NAME_DEFAULT
        )
        self.feature_extractor: nn.Module = CLIPModel.from_pretrained(pretrained_model_name_or_path, cache_dir=CACHE_DIR)
        self.projection_dim = self.feature_extractor.config.projection_dim
        self.discriminator = ClipDiscriminator(self.projection_dim, num_classes)
        self.discriminator_ = ClipDiscriminator(self.projection_dim, num_classes)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, pixel_values, input_ids, attention_mask):
        # Get image and text features
        image_features = self.feature_extractor.get_image_features(pixel_values=pixel_values)
        text_features = self.feature_extractor.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get final logits
        logits = self.discriminator(image_features, text_features)
        logits_2 = self.discriminator_(image_features, text_features)
        return logits, logits_2