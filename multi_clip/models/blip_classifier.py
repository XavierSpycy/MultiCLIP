from typing import Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BlipForImageTextRetrieval

from ..models.config import CACHE_DIR, BLIP_PRETRAINED_MODEL_NAME_DEFAULT, BLIP_DEFAULT_POOL_STRATEGY
from ..models.router import Router
from ..models.gat import GATLayer
from ..models.ml_decoder import MLDecoder

def get_multimodal_embeddings(
        vision_model,
        text_encoder,
        config,
        pool_strategy: str,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: bool = False,
        return_image_embeds: bool = False,
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
    
    vision_outputs = vision_model(
        pixel_values=pixel_values, 
        return_dict=config.use_return_dict,
        output_attentions=config.output_attentions,
        output_hidden_states=config.output_hidden_states,
        interpolate_pos_encoding=interpolate_pos_encoding,
    )

    if pool_strategy == 'cls_token':
        image_embeds: torch.Tensor = vision_outputs[0]
    else:
        image_embeds: torch.Tensor = vision_outputs.last_hidden_state
    
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

    text_outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=config.use_return_dict,
    )
    
    multimodal_embeds = text_outputs[0] if not config.use_return_dict else text_outputs.last_hidden_state

    if return_image_embeds:
        return image_embeds, multimodal_embeds
    else:
        return multimodal_embeds

def valid_pool_strategy(pool_strategy: str) -> None:
    if pool_strategy not in ['cls_token', 'last_hidden_state']:
        raise ValueError(f"pool_strategy must be one of ['cls_token', 'last_hidden_state'], got {pool_strategy}")

class BlipClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int) -> None:
        super(BlipClassificationHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
    
    def forward(self, multimodal_features: torch.FloatTensor) -> torch.FloatTensor:
        logits = self.head(multimodal_features)
        return logits
    
class BlipClassifier(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 pretrained_model_name_or_path: Optional[str] = None,
                 pool_strategy: Literal['cls_token', 'last_hidden_state'] = BLIP_DEFAULT_POOL_STRATEGY) -> None:
        
        super(BlipClassifier, self).__init__()

        valid_pool_strategy(pool_strategy)
        self.pool_strategy = pool_strategy

        pretrained_model_name_or_path = (
            pretrained_model_name_or_path if pretrained_model_name_or_path is not None else BLIP_PRETRAINED_MODEL_NAME_DEFAULT
        )
        base_model = BlipForImageTextRetrieval.from_pretrained(pretrained_model_name_or_path, cache_dir=CACHE_DIR)
        self.projection_dim = base_model.config.text_config.hidden_size
        self.config = base_model.config

        self.vision_model: nn.Module = base_model.vision_model
        self.text_encoder: nn.Module = base_model.text_encoder
        self.cls_head = BlipClassificationHead(self.projection_dim, num_classes)
        
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def get_multimodal_embeddings(
            self, 
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            interpolate_pos_encoding: bool = False,
        ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        
        return get_multimodal_embeddings(
            self.vision_model, self.text_encoder, self.config, self.pool_strategy, 
            input_ids, pixel_values, attention_mask, interpolate_pos_encoding)
    
    def get_multimodal_features(
            self, 
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            interpolate_pos_encoding: bool = False,
        ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        
        multimodal_embeds = self.get_multimodal_embeddings(
            input_ids=input_ids, 
            pixel_values=pixel_values, 
            attention_mask=attention_mask,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        multimodal_features = multimodal_embeds[:, 0, :]
        return multimodal_features
    
    def get_logits_from_features(self, multimodal_features: torch.FloatTensor) -> torch.FloatTensor:
        logits = self.cls_head(multimodal_features)
        return logits
    
    def forward(self, pixel_values, input_ids, attention_mask):
        multimodal_features = self.get_multimodal_features(input_ids, pixel_values, attention_mask)
        logits = self.cls_head(multimodal_features)
        return logits

class BlipEnsembleClassifier(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 pretrained_model_name_or_path: Optional[str] = None,
                 pool_strategy: Literal['cls_token', 'last_hidden_state'] = BLIP_DEFAULT_POOL_STRATEGY) -> None:
        
        super(BlipEnsembleClassifier, self).__init__()

        valid_pool_strategy(pool_strategy)
        self.pool_strategy = pool_strategy

        pretrained_model_name_or_path = (
            pretrained_model_name_or_path if pretrained_model_name_or_path is not None else BLIP_PRETRAINED_MODEL_NAME_DEFAULT
        )
        base_model = BlipForImageTextRetrieval.from_pretrained(pretrained_model_name_or_path, cache_dir=CACHE_DIR)
        self.projection_dim = base_model.config.text_config.hidden_size
        self.config = base_model.config

        self.vision_model: nn.Module = base_model.vision_model
        self.text_encoder: nn.Module = base_model.text_encoder
        self.vision_proj: nn.Module = base_model.vision_proj
        self.text_proj: nn.Module = base_model.text_proj
        self.cls_head_image = BlipClassificationHead(self.projection_dim, num_classes)
        self.cls_head_text = BlipClassificationHead(self.projection_dim, num_classes)
        self.cls_head_multimodal = BlipClassificationHead(self.projection_dim, num_classes)
        self.router = Router(self.projection_dim, 1)
    
    def get_image_features(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        if self.pool_strategy == 'cls_token':
            image_embeds = vision_outputs[0]
        else:
            image_embeds = vision_outputs.last_hidden_state
        image_features = self.vision_proj(image_embeds[:, 0, :])
        image_features = F.normalize(image_features, dim=-1)
        return image_features
    
    def get_text_features(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> torch.FloatTensor:
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.pool_strategy == 'cls_token':
            text_embeds = text_outputs[0]
        else:
            text_embeds = text_outputs.last_hidden_state
        text_features = self.text_proj(text_embeds[:, 0, :])
        text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    def get_multimodal_features(
            self, 
            input_ids: Optional[torch.LongTensor] = None, 
            pixel_values: Optional[torch.FloatTensor] = None, 
            attention_mask: Optional[torch.Tensor] = None, 
            interpolate_pos_encoding: bool = False, 
            return_image_features: Optional[bool] = True,):
        
        image_embeds, multimodal_embeds = get_multimodal_embeddings(
            self.vision_model, self.text_encoder, self.config, self.pool_strategy,
            input_ids, pixel_values, attention_mask, interpolate_pos_encoding, 
            return_image_embeds=return_image_features
        )

        multimodal_features = multimodal_embeds[:, 0, :]
        multimodal_features = F.normalize(self.text_proj(multimodal_features), dim=-1)

        if return_image_features:
            image_feautres = F.normalize((self.vision_proj(image_embeds[:, 0, :])), dim=-1)
            return image_feautres, multimodal_features
        else:
            return multimodal_features

    def forward(self, pixel_values, input_ids, attention_mask):
        image_feautures, multimodal_features = self.get_multimodal_features(input_ids, pixel_values, attention_mask)
        text_features = self.get_text_features(input_ids, attention_mask)

        image_logits = self.cls_head_image(image_feautures)
        text_logits = self.cls_head_text(text_features)
        multimodal_logits = self.cls_head_multimodal(multimodal_features)

        naive_image_weight = self.router(image_feautures)
        naive_text_weight = self.router(text_features)
        naive_multimodal_weight = self.router(multimodal_features)
        weights = torch.softmax(torch.cat([naive_image_weight, naive_text_weight, naive_multimodal_weight], dim=1), dim=1)
        image_weight, text_weight, multimodal_weight = weights[:, 0], weights[:, 1], weights[:, 2]

        logits = image_weight[:, None] * image_logits + text_weight[:, None] * text_logits + multimodal_weight[:, None] * multimodal_logits
        return logits

class BlipBoostClassifier(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 pretrained_model_name_or_path: Optional[str] = None,
                 pool_strategy: Literal['cls_token', 'last_hidden_state'] = BLIP_DEFAULT_POOL_STRATEGY) -> None:
        
        super(BlipBoostClassifier, self).__init__()

        valid_pool_strategy(pool_strategy)
        self.pool_strategy = pool_strategy

        pretrained_model_name_or_path = (
            pretrained_model_name_or_path if pretrained_model_name_or_path is not None else BLIP_PRETRAINED_MODEL_NAME_DEFAULT
        )
        base_model = BlipForImageTextRetrieval.from_pretrained(pretrained_model_name_or_path, cache_dir=CACHE_DIR)
        self.projection_dim = base_model.config.text_config.hidden_size
        self.config = base_model.config

        self.vision_model: nn.Module = base_model.vision_model
        self.text_encoder: nn.Module = base_model.text_encoder
        self.cls_head_1 = BlipClassificationHead(self.projection_dim, num_classes)
        self.cls_head_2 = BlipClassificationHead(self.projection_dim, num_classes)
    
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def get_multimodal_features(
            self, 
            input_ids: Optional[torch.LongTensor] = None, 
            pixel_values: Optional[torch.FloatTensor] = None, 
            attention_mask: Optional[torch.Tensor] = None, 
            interpolate_pos_encoding: bool = False,):
        
        return get_multimodal_embeddings(
            self.vision_model, self.text_encoder, self.config, self.pool_strategy,
            input_ids, pixel_values, attention_mask, interpolate_pos_encoding)[:, 0, :]
        
    def forward(self, pixel_values, input_ids, attention_mask):
        multimodal_features = self.get_multimodal_features(input_ids, pixel_values, attention_mask)
        logits = self.cls_head_1(multimodal_features)
        residuals = self.cls_head_2(multimodal_features)
        return torch.sigmoid(logits) + residuals

class BlipGATClassifier(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 pretrained_model_name_or_path: Optional[str] = None,
                 pool_strategy: Literal['cls_token', 'last_hidden_state'] = 'cls_token') -> None:
        
        super(BlipGATClassifier, self).__init__()

        valid_pool_strategy(pool_strategy)
        self.pool_strategy = pool_strategy

        pretrained_model_name_or_path = (
            pretrained_model_name_or_path if pretrained_model_name_or_path is not None else BLIP_PRETRAINED_MODEL_NAME_DEFAULT
        )
        base_model = BlipForImageTextRetrieval.from_pretrained(pretrained_model_name_or_path, cache_dir=CACHE_DIR)
        self.projection_dim = base_model.config.text_config.hidden_size
        self.config = base_model.config

        self.vision_model: nn.Module = base_model.vision_model
        self.text_encoder: nn.Module = base_model.text_encoder
        self.gat_layer = GATLayer(self.projection_dim)
        self.cls_heads = nn.ModuleList([BlipClassificationHead(self.projection_dim, 1) for _ in range(num_classes)])
        
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def get_multimodal_features(
            self, 
            input_ids: Optional[torch.LongTensor] = None, 
            pixel_values: Optional[torch.FloatTensor] = None, 
            attention_mask: Optional[torch.Tensor] = None, 
            interpolate_pos_encoding: bool = False,):
        
        return get_multimodal_embeddings(
            self.vision_model, self.text_encoder, self.config, self.pool_strategy,
            input_ids, pixel_values, attention_mask, interpolate_pos_encoding)[:, 0, :]

    def forward(self, pixel_values, input_ids, attention_mask, edge_index=None, edge_weight=None):
        multimodal_features = self.get_multimodal_features(input_ids, pixel_values, attention_mask)
        if edge_index is not None:
            gat_features = self.gat_layer(multimodal_features, edge_index, edge_weight=edge_weight)
        else:
            gat_features = multimodal_features
        logits = torch.cat([cls_head(gat_features) for cls_head in self.cls_heads], dim=1)
        return logits

class BlipMLDecoderClassifier(nn.Module):
    def __init__(self, 
                 num_classes: int, 
                 pretrained_model_name_or_path: Optional[str] = None,
                 pool_strategy: Literal['cls_token', 'last_hidden_state'] = 'cls_token') -> None:
        
        super(BlipMLDecoderClassifier, self).__init__()

        valid_pool_strategy(pool_strategy)
        self.pool_strategy = pool_strategy

        pretrained_model_name_or_path = (
            pretrained_model_name_or_path if pretrained_model_name_or_path is not None else BLIP_PRETRAINED_MODEL_NAME_DEFAULT
        )
        base_model = BlipForImageTextRetrieval.from_pretrained(pretrained_model_name_or_path, cache_dir=CACHE_DIR)
        self.vision_model: nn.Module = base_model.vision_model
        self.text_encoder: nn.Module = base_model.text_encoder
        self.config = base_model.config
        self.projection_dim = self.config.text_config.hidden_size
        self.ml_decoder = MLDecoder(num_classes, initial_num_features=self.projection_dim)

        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def get_multimodal_embeddings(
            self, 
            input_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            interpolate_pos_encoding: bool = False):
        
        return get_multimodal_embeddings(
            self.vision_model, self.text_encoder, self.config, self.pool_strategy,
            input_ids, pixel_values, attention_mask, interpolate_pos_encoding)
    
    def forward(self, pixel_values, input_ids, attention_mask):
        multimodal_embeddings = self.get_multimodal_embeddings(input_ids, pixel_values, attention_mask)
        logits = self.ml_decoder(multimodal_embeddings)
        return logits