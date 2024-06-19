from .utils import LabelEncoder
from .utils.tools import (
    seed_everything,
    read_csv,
    load_data,
    train_test_split,
    train_test_split_with_embeds,
    get_model_size,
    get_trainable_blocks_size,
    get_frozen_blocks_size,
    total_params_count,
    trainable_params_count,
    frozen_params_count,
    print_logging_info,
    plot_history,
    get_edge_relationship,
    parse_checkpoint_path)
from .utils.losses import (
    compute_binary_cross_entropy_loss_with_logits, 
    compute_zlpr_loss_with_logits, 
    compute_angular_additive_margin_loss_with_logits)
from .utils.predict_func import default_compute_predict_proba
from .utils.inference_func import (
    inference,
    get_best_thresholds,
    predict_on_test_set)
from .datasets import BlipDataset, ClipDataset
from .models.config import (
    BLIP_PRETRAINED_MODEL_NAME_DEFAULT, 
    CLIP_PRETRAINED_MODEL_NAME_DEFAULT,
    BLIP_PRETRAINED_MODEL_NAME_BASE,
    CLIP_PRETRAINED_MODEL_NAME_BASE)
from .models import (
    BlipClassifier, 
    BlipMLDecoderClassifier,
    ClipClassifier)
from .models.clip_classifier import ClipDiscriminator
from .trainers import (
    TrainArguments, 
    Trainer, 
    BoostTrainArguments, 
    BoostTrainer,
    ClipTrainArguments,
    ClipTrainer,
    HeadTrainArguments,
    HeadTrainer,
    MLDTrainer)
from .trainers.base_trainer import default_compute_outputs
from .trainers.ml_decoder_trainer import compute_mldecoder_outputs

ARGS_TO_SETTING = {
    "blip": {
        "dataset": BlipDataset,
        "model_type": BlipClassifier,
        "model_path": {
            "base":  BLIP_PRETRAINED_MODEL_NAME_BASE,
            "large": BLIP_PRETRAINED_MODEL_NAME_DEFAULT},
        "trainer": Trainer,
        "train_args": TrainArguments,
    },
    "blip_ml_decoder": {
        "dataset": BlipDataset,
        "model_type": BlipMLDecoderClassifier,
        "model_path": {
            "base":  BLIP_PRETRAINED_MODEL_NAME_BASE,
            "large": BLIP_PRETRAINED_MODEL_NAME_DEFAULT},
        "trainer": MLDTrainer,
        "train_args": TrainArguments,
    },
    "clip": {
        "dataset": ClipDataset,
        "model_type": ClipClassifier,
        "model_path": {
            "base":  CLIP_PRETRAINED_MODEL_NAME_BASE,
            "large": CLIP_PRETRAINED_MODEL_NAME_DEFAULT},
        "trainer": Trainer,
        "train_args": TrainArguments,
    },
    "clip_boost":{
        "dataset": ClipDataset,
        "model_type": ClipDiscriminator,
        "pretrained_model_type": ClipClassifier,
        "model_path": {
            "base":  CLIP_PRETRAINED_MODEL_NAME_BASE,
            "large": CLIP_PRETRAINED_MODEL_NAME_DEFAULT},
        "trainer": BoostTrainer,
        "train_args": BoostTrainArguments,
    },
    "clip_two_step":{
        "dataset": ClipDataset,
        "model_type": ClipClassifier,
        "model_path": {
            "base":  CLIP_PRETRAINED_MODEL_NAME_BASE,
            "large": CLIP_PRETRAINED_MODEL_NAME_DEFAULT},
        "trainer": ClipTrainer,
        "train_args": ClipTrainArguments,
    },
    "clip_head":{
        "dataset": ClipDataset,
        "model_type": ClipClassifier,
        "model_path": {
            "base":  CLIP_PRETRAINED_MODEL_NAME_BASE,
            "large": CLIP_PRETRAINED_MODEL_NAME_DEFAULT},
        "trainer": HeadTrainer,
        "train_args": HeadTrainArguments,
    },
}

ARGS_TO_LOSS_FUNC = {
    "bce": compute_binary_cross_entropy_loss_with_logits,
    "zlpr": compute_zlpr_loss_with_logits,
    "aam": compute_angular_additive_margin_loss_with_logits,
}

__all__ = [
    "LabelEncoder",
    "seed_everything",
    "read_csv",
    "load_data",
    "train_test_split",
    "train_test_split_with_embeds",
    "get_model_size",
    "get_trainable_blocks_size",
    "get_frozen_blocks_size",
    "total_params_count",
    "trainable_params_count",
    "frozen_params_count",
    "print_logging_info",
    "plot_history",
    "default_compute_predict_proba",
    "get_edge_relationship",
    "parse_checkpoint_path",
    "inference",
    "get_best_thresholds",
    "predict_on_test_set",
    "default_compute_outputs",
    "compute_mldecoder_outputs",
]

__version__ = "0.1.0"