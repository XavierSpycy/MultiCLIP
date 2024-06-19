import os
import argparse

import torch

import multi_clip
import multi_clip.utils as utils
from multi_clip import ARGS_TO_SETTING, ARGS_TO_LOSS_FUNC
from multi_clip import default_compute_outputs, compute_mldecoder_outputs

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=3407)
parser.add_argument("--num_epochs", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--model_name", type=str, default="blip")
parser.add_argument("--model_size", type=str, default="large")
parser.add_argument("--loss_name", type=str, default="bce")
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--greater_f1", action="store_true")
parser.add_argument("--version", type=str, default="v1")
parser.add_argument("--gpu_id", type=int, default=0)
args = parser.parse_args()

monitored_metric = "f1" if args.greater_f1 else "loss"
save_path_prefix = f"{args.model_name}_{args.model_size}" + \
        f"_{args.loss_name}_{args.version}" + \
            f"_lr{args.learning_rate}_bs{args.batch_size}" + \
                f"_seed{args.random_seed}_{monitored_metric}"

multi_clip.seed_everything(args.random_seed)
device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    alternative_device = torch.device(f"cuda:{1 - args.gpu_id}")

label_encoder = utils.LabelEncoder.from_pretrained()
image_paths_and_texts, labels = multi_clip.load_data('train', label_encoder=label_encoder)
(image_paths_train, texts_train, labels_train), (image_paths_val, texts_val, labels_val) = multi_clip.train_test_split(
    image_paths_and_texts, labels, test_size=0.15)

setting = ARGS_TO_SETTING[args.model_name]
pretrained_model_name_or_path = setting["model_path"][args.model_size]
dataset_obj = setting["dataset"]
model_type_obj = setting["model_type"]
trainer_obj = setting["trainer"]
train_args_obj = setting["train_args"]

model = model_type_obj(
    num_classes=1 if args.model_name == "clip_head" else len(label_encoder.classes_), 
    pretrained_model_name_or_path=pretrained_model_name_or_path).to(device)

if args.model_name == "blip_ml_decoder":
    model.ml_decoder = model.ml_decoder.to(alternative_device)

train_dataset = dataset_obj(
    image_paths_train, texts_train, labels_train, 
    pretrained_model_name_or_path=pretrained_model_name_or_path)
val_dataset = dataset_obj(
    image_paths_val, texts_val, labels_val, 
    pretrained_model_name_or_path=pretrained_model_name_or_path)

train_args = train_args_obj(
    train_set=train_dataset,
    val_set=val_dataset,
    num_epochs=args.num_epochs,
    train_batch_size=args.batch_size,
    val_batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    checkpoint_best_model_path=save_path_prefix + ".pth",
    save_model=True,
    compute_loss=ARGS_TO_LOSS_FUNC[args.loss_name],
    early_stopping_by='metric' if args.greater_f1 else 'loss',
    is_greater_better=args.greater_f1,
    compute_outputs=compute_mldecoder_outputs if args.model_name == "blip_ml_decoder" else default_compute_outputs,
)

trainer = trainer_obj(model, train_args)
history = trainer.train()
multi_clip.plot_history(history, save_path=os.path.join("figures", save_path_prefix + ".png"))