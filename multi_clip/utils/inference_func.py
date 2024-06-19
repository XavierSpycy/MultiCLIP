from typing import Optional, Union, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from ..trainers.base_trainer import default_compute_outputs
from .predict_func import default_compute_predict_proba
from .tools import load_data
from .label_encoder import LabelEncoder

def inference(model: nn.Module, 
              data_set: Dataset, 
              device: Union[str, torch.device]='cpu', 
              batch_size=1,
              compute_outputs=default_compute_outputs,
              compute_predict_proba=default_compute_predict_proba):
    predict_probas = []
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model.to(device).eval()
    with torch.no_grad():
        for inputs, _ in tqdm(data_loader):
            model_outputs = compute_outputs(inputs, model, device)
            predict_proba = compute_predict_proba(model_outputs)
            predict_probas.append(predict_proba.cpu().numpy())
    predict_probas = np.concatenate(predict_probas, axis=0)
    return predict_probas

def get_best_thresholds(predict_probas: np.ndarray, 
                        labels, 
                        thresholds=None):
    
    num_classes = predict_probas.shape[1]
    if thresholds is None:
        thresholds = np.ones((num_classes, )) * 0.5
    else:
        thresholds = np.array(thresholds)
    
    for i in tqdm(range(num_classes)):
        best_candidate = None
        best_f1 = 0
        for candidate in np.linspace(0.01, 0.80, 80):
            thresholds[i] = candidate
            predictions = predict_probas > thresholds
            f1 = f1_score(labels, predictions, average='micro')
            if f1 > best_f1:
                best_f1 = f1
                best_candidate = candidate
        thresholds[i] = best_candidate
    return thresholds

def predict_on_test_set(model: nn.Module, 
                        data_set_object: Callable,
                        label_encoder: LabelEncoder, 
                        thresholds: Optional[np.ndarray] = None, 
                        batch_size: int = 1,
                        device: Union[str, torch.device] = 'cpu',
                        output_file: Optional[str] = None,
                        compute_predcit_proba=default_compute_predict_proba):
    if thresholds is None:
        thresholds = np.ones((len(label_encoder.classes_), )) * 0.5
    
    image_paths_and_texts_test, labels_test = load_data('test', label_encoder=label_encoder)
    image_paths_test, texts_test = image_paths_and_texts_test[:, 0], image_paths_and_texts_test[:, 1]

    data_set = data_set_object(image_paths_test, texts_test, labels_test)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    prediction_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            model_outputs = default_compute_outputs(inputs, model, device)
            predict_proba = compute_predcit_proba(model_outputs)
            prediction_probs.append(predict_proba.cpu().numpy())
    
    prediction_probs = np.vstack(prediction_probs)
    predictions = (prediction_probs > thresholds).astype(int)

    for i in range(len(predictions)):
        if not predictions[i].any():  # Check if all values are zero
            # Find the label(s) closest to their respective thresholds
            distance_to_threshold = np.abs(prediction_probs[i] - thresholds)
            min_distance = np.min(distance_to_threshold)
            predictions[i] = (distance_to_threshold == min_distance).astype(int)

    decoded_predictions = label_encoder.decode(predictions)

    if output_file is not None:
        submission = pd.DataFrame({'ImageID': image_paths_test, 'Labels': decoded_predictions})
        submission['ImageID'] = submission['ImageID'].apply(lambda x: x.split('/')[-1])
        submission.to_csv(output_file, index=False)
    
    return decoded_predictions