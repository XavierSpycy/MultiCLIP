import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from tqdm import tqdm

from .. import print_logging_info
from ..utils.metrics import precision_score_binary, recall_score_binary
from ..utils.losses import compute_binary_focal_loss_with_logits
from ..models.clip_classifier import ClipClassifier
from .base_trainer import default_compute_outputs

logging.basicConfig(level=logging.INFO)

def default_lr_scheduler_kwargs():
    return {"mode": "min", "factor": 0.85, "patience": 1, "verbose": True}

def default_kwargs():
    return {}

@dataclass
class HeadTrainArguments:
    train_set: Dataset
    val_set: Dataset
    do_eval: bool = True
    class_index: int = 1
    num_epochs: int = 10
    train_batch_size: int = 32
    val_batch_size: int = 32
    optimizer: Optimizer = AdamW
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    lr_scheduler: Optional[_LRScheduler] = None
    lr_scheduler_kwargs: dict = field(default_factory=default_lr_scheduler_kwargs)
    compute_outputs: Callable = field(default=default_compute_outputs)
    output_kwargs: dict = field(default_factory=default_kwargs)
    compute_loss: Callable = field(default=compute_binary_focal_loss_with_logits)
    loss_kwargs: dict = field(default_factory=default_kwargs)
    eval_delay: int = 0
    eval_epochs: int = 1
    metric_name_for_best_model_1: str = "precision"
    metric_for_best_model_1: Callable = precision_score_binary
    metric_name_for_best_model_2: str = "recall"
    metric_for_best_model_2: Callable = recall_score_binary
    load_best_model_at_end: bool = True
    early_stopping_patience: int = 10
    amp: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    checkpoint_dir: Optional[str] = "checkpoints"
    checkpoint_best_model_path: Optional[str] = None
    save_model: Optional[bool] = None

class HeadTrainer:
    def __init__(self, model: ClipClassifier, train_args: HeadTrainArguments):
        self.model = model
        self.train_args = train_args
        
    def train(self):
        metric_name_for_best_model_1 = self.train_args.metric_name_for_best_model_1
        metric_name_for_best_model_2 = self.train_args.metric_name_for_best_model_2

        history = {}
        history["train_loss"] = []
        history["val_loss"] = []
        history[f"train_{metric_name_for_best_model_1}"] = []
        history[f"val_{metric_name_for_best_model_1}"] = []
        history[f"train_{metric_name_for_best_model_2}"] = []
        history[f"val_{metric_name_for_best_model_2}"] = []
        logging_metric_name_1 = self.train_args.metric_name_for_best_model_1.replace("_", " ").capitalize()
        logging_metric_name_2 = self.train_args.metric_name_for_best_model_2.replace("_", " ").capitalize()
        
        class_index = self.train_args.class_index
        optimizer: Optimizer = self.train_args.optimizer(
            self.model.parameters(), 
            lr=self.train_args.learning_rate, 
            betas=(self.train_args.adam_beta1, self.train_args.adam_beta2),
            eps=self.train_args.adam_eps,
            weight_decay=self.train_args.weight_decay)
        scheduler: _LRScheduler = self.train_args.lr_scheduler(optimizer, **self.train_args.lr_scheduler_kwargs) if self.train_args.lr_scheduler is not None else None
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
            train_loss, train_metric_1, train_metric_2 = do_train(
                class_index, 
                self.model, train_loader, optimizer, compute_outputs, 
                output_kwargs, compute_loss, loss_kwargs, 
                self.train_args.metric_for_best_model_1, self.train_args.metric_for_best_model_2, 
                scaler)
            
            history["train_loss"].append(train_loss)
            history[f"train_{metric_name_for_best_model_1}"].append(train_metric_1)
            history[f"train_{metric_name_for_best_model_2}"].append(train_metric_2)

            tqdm.write(f"Epoch: {epoch+1}/{self.train_args.num_epochs}, Train Loss: {train_loss:.4f}, Train {logging_metric_name_1}: {train_metric_1:.4f}, Train {logging_metric_name_2}: {train_metric_2:.4f}")

            if self.train_args.do_eval:
                if (epoch + 1) % self.train_args.eval_epochs == 0:
                    val_loss, val_metric_1, val_metric_2 = do_eval(
                        class_index,
                        self.model, val_loader, compute_outputs, 
                        output_kwargs, compute_loss, loss_kwargs, 
                        self.train_args.metric_for_best_model_1, 
                        self.train_args.metric_for_best_model_2)
                    
                    if scheduler is not None:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(val_loss)
                        else:
                            scheduler.step()
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
                    if early_stopping_counter == self.train_args.early_stopping_patience:
                        logging.info(f"Early stopping. Best epoch: {best_epoch+1}/{self.train_args.num_epochs}.")
                        break

                    history["val_loss"].append(val_loss)
                    history[f"val_{metric_name_for_best_model_1}"].append(val_metric_1)
                    history[f"val_{metric_name_for_best_model_2}"].append(val_metric_2)
                    tqdm.write(f"Epoch: {epoch+1}/{self.train_args.num_epochs}, Val Loss: {val_loss:.4f}, Val {logging_metric_name_1}: {val_metric_1:.4f}, Val {logging_metric_name_2}: {val_metric_2:.4f}")
        
        if self.train_args.load_best_model_at_end:
            self.model.load_state_dict(best_model_weight)
        return history

def do_train(
        class_index: int,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        compute_outputs: Callable,
        output_kwargs: Dict,
        compute_loss: Callable,
        loss_kwargs: Dict,
        metric_for_best_model_1: Callable,
        metric_for_best_model_2: Callable,
        scaler: Optional[GradScaler] = None) -> Tuple[float, float]:
    
    all_outputs = []
    all_labels = []

    train_loss = 0
    train_metric_1 = 0
    train_metric_2 = 0
    train_num_samples = 0
    model.train()
    device = next(model.parameters()).device

    for inputs, labels in train_loader:
        batch_size = labels.size(0)
        labels = labels[:, class_index].unsqueeze(1).to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler is not None):
            outputs = compute_outputs(inputs, model, device, **output_kwargs)
            loss: torch.Tensor = compute_loss(outputs, labels, **loss_kwargs)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        train_loss += loss.item() * batch_size
        train_num_samples += batch_size

        all_labels.append(labels.detach().cpu())
        all_outputs.append(outputs.detach().cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    train_loss = train_loss / train_num_samples
    train_metric_1 = metric_for_best_model_1(all_outputs, all_labels)
    train_metric_2 = metric_for_best_model_2(all_outputs, all_labels)
    return train_loss, train_metric_1, train_metric_2

def do_eval(
        class_index: int,
        model: nn.Module, 
        val_loader: DataLoader, 
        compute_outputs: Callable, 
        ouput_kwargs: Dict,
        compute_loss: Callable, 
        loss_kwargs: Dict, 
        metric_for_best_model_1: Callable,
        metric_for_best_model_2: Callable) -> Tuple[float, float, float]:
    
    model.eval()
    device = next(model.parameters()).device
    val_loss = 0
    val_metric_1 = 0
    val_metric_2 = 0
    val_num_samples = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            batch_size = labels.size(0)
            labels = labels[:, class_index].unsqueeze(1).to(device)
            outputs = compute_outputs(inputs, model, device, **ouput_kwargs)
            loss: torch.FloatTensor = compute_loss(outputs, labels, **loss_kwargs)
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            val_loss += loss.item() * batch_size
            val_num_samples += batch_size
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    val_loss = val_loss / val_num_samples
    val_metric_1 = metric_for_best_model_1(all_outputs, all_labels)
    val_metric_2 = metric_for_best_model_2(all_outputs, all_labels)
    return val_loss, val_metric_1, val_metric_2