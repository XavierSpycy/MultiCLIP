import argparse

import torch

from multi_clip import (
    LabelEncoder,
    ARGS_TO_SETTING,
    load_data,
    train_test_split,
    inference, 
    get_best_thresholds, 
    predict_on_test_set, 
    parse_checkpoint_path,
    compute_mldecoder_outputs,
    default_compute_outputs)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--gpu_id", type=int, default=0)
args = parser.parse_args()

checkpoint_path = args.checkpoint_path

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    alternative_device = torch.device(f"cuda:{1 - args.gpu_id}")

output_file = "submission_" + checkpoint_path.split("/")[-1].split(".pth")[0] + ".csv"

label_encoder = LabelEncoder.from_pretrained()
image_paths_and_texts, labels = load_data('train', label_encoder=label_encoder)
(image_paths_train, texts_train, labels_train), (image_paths_val, texts_val, labels_val) = train_test_split(
    image_paths_and_texts, labels, test_size=0.2)

model_name, model_size = parse_checkpoint_path(checkpoint_path)
setting = ARGS_TO_SETTING[model_name]
pretrained_model_name_or_path = setting["model_path"][model_size]
dataset_obj = setting["dataset"]
model_type_obj = setting["model_type"]

num_classes = len(label_encoder.classes_)
model = model_type_obj(
    num_classes=1 if model_name == "clip_head" else len(label_encoder.classes_), 
    pretrained_model_name_or_path=pretrained_model_name_or_path).to(device)

if model_name == "blip_ml_decoder":
    model.ml_decoder = model.ml_decoder.to(alternative_device)

model.load_state_dict(torch.load(checkpoint_path))
val_dataset = dataset_obj(image_paths_val, texts_val, labels_val, pretrained_model_name_or_path=pretrained_model_name_or_path)

logits = inference(
    model, 
    val_dataset, 
    device=device, 
    batch_size=args.batch_size,
    compute_outputs=compute_mldecoder_outputs if model_name == "blip_ml_decoder" else default_compute_outputs)

thresholds = get_best_thresholds(logits, labels_val)

predict_on_test_set(
    model, dataset_obj, label_encoder, thresholds=thresholds, device=device, 
    batch_size=args.batch_size, output_file=output_file)