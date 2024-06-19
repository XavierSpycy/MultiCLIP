import os
import re
import random
import logging
from io import StringIO, BytesIO
from typing import Dict, List, Optional, Tuple, Union, Literal, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.typing import ColorType
from skmultilearn.model_selection import iterative_train_test_split

from .label_encoder import LabelEncoder

logging.basicConfig(level=logging.INFO)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_csv(path: Union[str, os.PathLike]) -> pd.DataFrame:
    with open(path) as file:
        lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    return pd.read_csv(StringIO(''.join(lines)), escapechar="/")

def load_data(
        partition: Literal['train', 'test'], 
        label_encoder: LabelEncoder, 
        return_label_only: bool=False) -> Union[pd.Series, Tuple[pd.DataFrame, pd.Series]]:
    
    if partition not in ["train", "test"]:
        raise ValueError("Invalid partition. Must be one of 'train', or 'test'.")
    
    df = read_csv(f'{partition}.csv')
    df['ImageID'] = df['ImageID'].apply(lambda x: os.path.join('data', x))
    if partition == "train":
        labels = df['Labels'].apply(lambda x: list(map(int, x.split(" "))))
        labels = label_encoder.encode(labels)
    else:
        labels = [[0] * 18 for _ in range(len(df))]
    image_paths_and_texts = df[['ImageID', 'Caption']].to_numpy()
    labels = np.array(labels)
    if return_label_only:
        return labels
    else:
        return image_paths_and_texts, labels

def train_test_split(
        image_paths_and_texts: pd.DataFrame, 
        labels: pd.Series, 
        test_size: float=0.1) -> Tuple[Tuple[pd.Series, pd.Series, pd.Series], Tuple[pd.Series, pd.Series, pd.Series]]:
    image_paths_and_texts_train, labels_train, image_paths_and_texts_val, labels_val = iterative_train_test_split(
        X=image_paths_and_texts, y=labels, test_size=test_size)
    image_paths_train, texts_train = image_paths_and_texts_train[:, 0], image_paths_and_texts_train[:, 1]
    image_paths_val, texts_val = image_paths_and_texts_val[:, 0], image_paths_and_texts_val[:, 1]
    return (image_paths_train, texts_train, labels_train), (image_paths_val, texts_val, labels_val)

def train_test_split_with_embeds(embeddings, labels, test_size=0.1):
    embeddings_train, labels_train, embeddings_val, labels_val = iterative_train_test_split(
        X=embeddings, y=labels, test_size=test_size)
    return (embeddings_train, labels_train), (embeddings_val, labels_val)

def get_model_size(model: nn.Module):
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer, _use_new_zipfile_serialization=True)
    size = buffer.tell() / (1024**2)
    buffer.close()
    return size

def get_trainable_blocks_size(model: nn.Module):
    buffer = BytesIO()
    trainable_params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    torch.save(trainable_params, buffer, _use_new_zipfile_serialization=True)
    size = buffer.tell() / (1024**2)
    buffer.close()
    return size

def get_frozen_blocks_size(model: nn.Module):
    buffer = BytesIO()
    frozen_params = {k: v for k, v in model.named_parameters() if not v.requires_grad}
    torch.save(frozen_params, buffer, _use_new_zipfile_serialization=True)
    size = buffer.tell() / (1024**2)
    buffer.close()
    return size

def total_params_count(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())

def trainable_params_count(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def frozen_params_count(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)

def print_logging_info(model: nn.Module):
    logging.info(f"The size of the model: {get_model_size(model):.2f} MB.")
    logging.info(f"The size of the trainable blocks: {get_trainable_blocks_size(model):.2f} MB.")
    logging.info(f"The number of total parameters: {total_params_count(model):,}.")
    logging.info(f"The number of trainable parameters: {trainable_params_count(model):,}.")

def plot_history(history: Dict[str, List], save_path: Optional[str]=None):
        def plot_curve(
                ax: Axes, 
                history: Dict[str, List], 
                metric_name: str, 
                train_color: Union[ColorType, Sequence[ColorType]], 
                val_color: Union[ColorType, Sequence[ColorType]]):
            ax.plot(history[f"train_{metric_name}"], label=f"Train {metric_name.capitalize()}", color=train_color)
            ax.plot(history[f"val_{metric_name}"], label=f"Validation {metric_name.capitalize()}", color=val_color)
            ax.legend()
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name.capitalize()} Curve")
        
        num_plots = len(history) // 2
        _, axs = plt.subplots(num_plots, 1, figsize=(8, 6 * num_plots))
        if num_plots == 1:
            axs = [axs]
        
        plot_curve(axs[0], history, "loss", "red", "orange")
        for key in history.keys():
            if not key.endswith("loss"):
                 if key.startswith("train"):
                     custom_metric_name = key.split("train_")[1]
                     break

        if num_plots == 2:
            plot_curve(axs[1], history, custom_metric_name, "blue", "green")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

def get_edge_relationship(labels: np.ndarray, normalize: bool=True, remove_diagonal: bool=True):
    num_classes = labels.shape[1]
    cooccurrence_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    for label in labels:
        indices = np.where(label)[0]
        for i in indices:
            for j in indices:
                cooccurrence_matrix[i, j] += 1
                
    if remove_diagonal:
        np.fill_diagonal(cooccurrence_matrix, 0)
        
    if normalize:
        row_sums = cooccurrence_matrix.sum(axis=1)
        cooccurrence_matrix[row_sums > 0] /= row_sums[row_sums > 0][:, np.newaxis]
    
    cooccurrence_matrix = torch.from_numpy(cooccurrence_matrix).float()
    edge_index = cooccurrence_matrix.nonzero(as_tuple=False).t().contiguous()
    edge_weight = cooccurrence_matrix[edge_index[0], edge_index[1]]
    
    return edge_index, edge_weight

def parse_checkpoint_path(checkpoint_path: Union[str, os.PathLike]) -> Tuple[str, str]:
    checkpoint_file_name = checkpoint_path.split("/")[-1]

    if 'large' in checkpoint_file_name:
        model_size = 'large'
    elif 'base' in checkpoint_file_name:
        model_size = 'base'
    else:
        raise ValueError("Invalid checkpoint path.")
    
    model_name = checkpoint_file_name.split("_" + model_size)[0]
    return model_name, model_size