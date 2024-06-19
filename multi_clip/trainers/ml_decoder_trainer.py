import os
import logging
from typing import Optional, Callable, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .. import print_logging_info
from ..models.blip_classifier import BlipMLDecoderClassifier
from .base_trainer import TrainArguments

logging.basicConfig(level=logging.INFO)

def default_lr_scheduler_kwargs():
    return {"mode": "min", "factor": 0.85, "patience": 1, "verbose": True}

def default_kwargs():
    return {}

def compute_mldecoder_outputs(
        inputs: Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
        model: BlipMLDecoderClassifier, 
        device_1: torch.device, 
        device_2: Optional[torch.device]=None, 
        **kwargs):
    pixel_values, input_ids, attention_mask = inputs
    pixel_values = pixel_values.to(device_1)
    input_ids = input_ids.to(device_1)
    attention_mask = attention_mask.to(device_1)
    if device_2 is not None:
        multimodal_embeddings = model.get_multimodal_embeddings(input_ids, pixel_values, attention_mask).to(device_2)
    else:
        multimodal_embeddings = model.get_multimodal_embeddings(input_ids, pixel_values, attention_mask)
    return model.ml_decoder(multimodal_embeddings)

class MLDTrainer:
    def __init__(self, model: BlipMLDecoderClassifier, train_args: TrainArguments):
        self.model = model
        self.train_args = train_args

    def train(self):
        metric_name_for_best_model = self.train_args.metric_name_for_best_model

        history = {}
        history["train_loss"] = []
        history["val_loss"] = []
        history[f"train_{metric_name_for_best_model}"] = []
        history[f"val_{metric_name_for_best_model}"] = []
        logging_metric_name = self.train_args.metric_name_for_best_model.replace("_", " ").capitalize()
        
        optimizer = self.train_args.optimizer(
            self.model.parameters(), 
            lr=self.train_args.learning_rate, 
            betas=(self.train_args.adam_beta1, self.train_args.adam_beta2),
            eps=self.train_args.adam_eps,
            weight_decay=self.train_args.weight_decay)
        scheduler = self.train_args.lr_scheduler(optimizer, **self.train_args.lr_scheduler_kwargs) if self.train_args.lr_scheduler is not None else None
        compute_loss = self.train_args.compute_loss
        loss_kwargs = self.train_args.loss_kwargs
        compute_outputs = self.train_args.compute_outputs
        output_kwargs = self.train_args.output_kwargs

        train_batch_size = self.train_args.train_batch_size
        val_batch_size = self.train_args.val_batch_size
        num_workers = self.train_args.dataloader_num_workers
        pin_memory = self.train_args.dataloader_pin_memory
        train_loader = DataLoader(self.train_args.train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(self.train_args.val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        min_val_loss = float("inf")
        min_val_metric = float("inf")
        max_val_metric = float("-inf")
        best_model_weight = None
        best_epoch = 0

        if self.train_args.checkpoint_dir is not None:
            os.makedirs(self.train_args.checkpoint_dir, exist_ok=True)
        else:
            self.train_args.checkpoint_dir = "checkpoints"
        
        checkpoint_dir = self.train_args.checkpoint_dir

        if self.train_args.checkpoint_best_model_path is not None:
            checkpoint_best_model = os.path.join(checkpoint_dir, self.train_args.checkpoint_best_model_path)
        else:
            checkpoint_best_model = os.path.join(checkpoint_dir, "best_model.pth")
        
        if self.train_args.early_stopping_patience > 0:
            early_stopping_counter = 0
        else:
            early_stopping_counter = None
        
        if self.train_args.amp:
            scaler = GradScaler()
        else:
            scaler = None
        
        print_logging_info(self.model)

        for epoch in tqdm(range(self.train_args.num_epochs)):
            train_loss, train_metric = do_train(self.model, train_loader, optimizer, compute_outputs, output_kwargs,
                                                compute_loss, loss_kwargs, self.train_args.metric_for_best_model, scaler)
            
            history["train_loss"].append(train_loss)
            history[f"train_{metric_name_for_best_model}"].append(train_metric)

            tqdm.write(f"Epoch: {epoch+1}/{self.train_args.num_epochs}, Train Loss: {train_loss:.4f}, Train {logging_metric_name}: {train_metric:.4f}")

            if self.train_args.do_eval:
                if (epoch + 1) % self.train_args.eval_epochs == 0:
                    val_loss, val_metric = do_eval(self.model, val_loader, compute_outputs, output_kwargs,
                                                   compute_loss, loss_kwargs, self.train_args.metric_for_best_model)
                    
                    if scheduler is not None:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(val_loss)
                        else:
                            scheduler.step()
                    
                    if self.train_args.early_stopping_by == "loss":
                        if val_loss < min_val_loss:
                            min_val_loss = val_loss
                            best_model_weight = self.model.state_dict()
                            if self.train_args.save_model:
                                torch.save(best_model_weight, checkpoint_best_model)
                            best_epoch = epoch
                            if early_stopping_counter is not None:
                                early_stopping_counter = 0
                        else:
                            if early_stopping_counter is not None:
                                early_stopping_counter += 1
                    elif self.train_args.early_stopping_by == "metric":
                        if self.train_args.is_greater_better:
                            if val_metric > max_val_metric:
                                max_val_metric = val_metric
                                best_model_weight = self.model.state_dict()
                                if self.train_args.save_model:
                                    torch.save(best_model_weight, checkpoint_best_model)
                                best_epoch = epoch
                                if early_stopping_counter is not None:
                                    early_stopping_counter = 0
                            else:
                                if early_stopping_counter is not None:
                                    early_stopping_counter += 1
                        else:
                            if val_metric < min_val_metric:
                                min_val_metric = val_metric
                                best_model_weight = self.model.state_dict()
                                if self.train_args.save_model:
                                    torch.save(best_model_weight, checkpoint_best_model)
                                best_epoch = epoch
                                if early_stopping_counter is not None:
                                    early_stopping_counter = 0
                            else:
                                if early_stopping_counter is not None:
                                    early_stopping_counter += 1
                    else:
                        raise ValueError("early_stop_by must be either 'loss' or 'metric'.")
                    
                    if early_stopping_counter == self.train_args.early_stopping_patience:
                        logging.info(f"Early stopping. Best epoch: {best_epoch+1}/{self.train_args.num_epochs}.")
                        break

                    history["val_loss"].append(val_loss)
                    history[f"val_{metric_name_for_best_model}"].append(val_metric)
                    tqdm.write(f"Epoch: {epoch+1}/{self.train_args.num_epochs}, Val Loss: {val_loss:.4f}, Val {logging_metric_name}: {val_metric:.4f}")
        
        if self.train_args.load_best_model_at_end:
            self.model.load_state_dict(best_model_weight)
        return history

def do_train(
        model: BlipMLDecoderClassifier,
        train_loader: DataLoader,
        optimizer: Optimizer,
        compute_outputs: Callable,
        output_kwargs: Dict,
        compute_loss: Callable,
        loss_kwargs: Dict,
        metric_for_best_model: Callable,
        scaler: Optional[GradScaler] = None) -> Tuple[float, float]:
    train_loss = 0
    train_metric = 0
    train_num_samples = 0
    model.train()
    device_1 = next(model.vision_model.parameters()).device
    device_2 = next(model.ml_decoder.parameters()).device

    for inputs, labels in train_loader:
        batch_size = labels.size(0)
        labels = labels.to(device_2)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler is not None):
            outputs = compute_outputs(inputs, model, device_1, device_2, **output_kwargs)
            loss = compute_loss(outputs, labels, **loss_kwargs)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        train_loss += loss.item() * batch_size
        train_metric += metric_for_best_model(outputs, labels) * batch_size
        train_num_samples += batch_size
    
    train_loss = train_loss / train_num_samples
    train_metric = train_metric / train_num_samples
    return train_loss, train_metric

def do_eval(
        model: BlipMLDecoderClassifier, 
        val_loader: DataLoader, 
        compute_outputs: Callable, 
        ouput_kwargs: Dict,
        compute_loss: Callable, 
        loss_kwargs: Dict, 
        metric_for_best_model: Callable) -> Tuple[float, float]:
    
    model.eval()
    device_1 = next(model.vision_model.parameters()).device
    device_2 = next(model.ml_decoder.parameters()).device
    val_loss = 0
    val_metric = 0
    val_num_samples = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            batch_size = labels.size(0)
            labels = labels.to(device_2)
            outputs = compute_outputs(inputs, model, device_1, device_2, **ouput_kwargs)
            loss = compute_loss(outputs, labels, **loss_kwargs)
            val_loss += loss.item() * batch_size
            val_metric += metric_for_best_model(outputs, labels) * batch_size
            val_num_samples += batch_size
    val_loss = val_loss / val_num_samples
    val_metric = val_metric / val_num_samples
    return val_loss, val_metric