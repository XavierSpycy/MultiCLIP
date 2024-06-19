import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Literal

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from tqdm import tqdm

from .. import print_logging_info
from ..utils.metrics import f1_score_micro
from ..utils.losses import compute_smooth_loss
from ..models import ClipClassifier
from .base_trainer import default_compute_outputs

logging.basicConfig(level=logging.INFO)

def default_lr_scheduler_kwargs():
    return {"mode": "min", "factor": 0.85, "patience": 1, "verbose": True}

def default_loss_kwargs():
    return {}

@dataclass
class ClipTrainArguments:
    train_set: Dataset
    val_set: Dataset
    do_eval: bool = True
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
    compute_loss: Callable = field(default=compute_smooth_loss)
    loss_kwargs: dict = field(default_factory=default_loss_kwargs)
    eval_delay: int = 0
    eval_epochs: int = 1
    metric_name_for_best_model: str = "f1_score_micro"
    metric_for_best_model: Callable = f1_score_micro
    load_best_model_at_end: bool = True
    early_stopping_by: Literal['loss', 'metric'] = "loss"
    is_greater_better: bool = False
    early_stopping_patience: int = 10
    amp: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    checkpoint_dir: Optional[str] = "checkpoints"
    checkpoint_best_model_path: Optional[str] = None
    save_model: Optional[bool] = None

class ClipTrainer:
    def __init__(self, model: ClipClassifier, train_args: ClipTrainArguments):
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
        
        device = next(self.model.parameters()).device

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

        do_warmup_image_cls = True
        do_warmup_text_cls = True
        image_metric_prev = 0
        text_metric_prev = 0

        for epoch in tqdm(range(self.train_args.num_epochs)):
            train_loss = 0
            train_metric = 0
            train_num_samples = 0
            image_metric_curr = 0
            text_metric_curr = 0

            self.model.train()
            for inputs, labels in train_loader:
                batch_size = labels.size(0)
                pixel_values, input_ids, attention_mask = inputs
                pixel_values = pixel_values.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)

                if do_warmup_image_cls:
                    with autocast(enabled=self.train_args.amp):
                        image_logits = self.model.get_image_logits_from_inputs(pixel_values)
                        image_loss: torch.Tensor = compute_loss(image_logits, labels, **loss_kwargs)
                    if self.train_args.amp:
                        scaler.scale(image_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        image_loss.backward()
                        optimizer.step()
                    
                    image_metric_curr += self.train_args.metric_for_best_model(image_logits, labels) * batch_size

                if do_warmup_text_cls:
                    with autocast(enabled=self.train_args.amp):
                        text_logits = self.model.get_text_logits_from_inputs(input_ids, attention_mask)
                        text_loss: torch.Tensor = compute_loss(text_logits, labels, **loss_kwargs)
                    if self.train_args.amp:
                        scaler.scale(text_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        text_loss.backward()
                        optimizer.step()
                    
                    text_metric_curr += self.train_args.metric_for_best_model(text_logits, labels) * batch_size
                
                if not do_warmup_image_cls and not do_warmup_text_cls:
                    with autocast(enabled=self.train_args.amp):
                        outputs = compute_outputs(inputs, self.model, device)
                        loss: torch.Tensor = compute_loss(outputs, labels, **loss_kwargs)
                    if self.train_args.amp:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                    
                    train_loss += loss.item() * batch_size
                    train_metric += self.train_args.metric_for_best_model(outputs, labels) * batch_size
                
                train_num_samples += batch_size
            
            if do_warmup_image_cls:
                image_metric_curr = image_metric_curr / train_num_samples
                if image_metric_curr - image_metric_prev < 0.01:
                    do_warmup_image_cls = False
                image_metric_prev = image_metric_curr
            if do_warmup_text_cls:
                text_metric_curr = text_metric_curr / train_num_samples
                if text_metric_curr - text_metric_prev < 0.01:
                    do_warmup_text_cls = False
                text_metric_prev = text_metric_curr
            if not do_warmup_image_cls and not do_warmup_text_cls:
                train_loss = train_loss / train_num_samples
                train_metric = train_metric / train_num_samples
                history["train_loss"].append(train_loss)
                history[f"train_{metric_name_for_best_model}"].append(train_metric)

                tqdm.write(f"Epoch: {epoch+1}/{self.train_args.num_epochs}, Train Loss: {train_loss:.4f}, Train {logging_metric_name}: {train_metric:.4f}")

            if self.train_args.do_eval and not do_warmup_image_cls and not do_warmup_text_cls:
                if (epoch + 1) % self.train_args.eval_epochs == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_loss = 0
                        val_metric = 0
                        val_num_samples = 0
                        for inputs, labels in val_loader:
                            batch_size = labels.size(0)
                            labels = labels.to(device)
                            outputs = compute_outputs(inputs, self.model, device)
                            loss = compute_loss(outputs, labels, **loss_kwargs)
                            val_loss += loss.item() * batch_size
                            val_metric += self.train_args.metric_for_best_model(outputs, labels) * batch_size
                            val_num_samples += batch_size
                    val_loss = val_loss / val_num_samples
                    val_metric = val_metric / val_num_samples

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