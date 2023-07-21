import os
import math
from argparse import ArgumentParser

import wandb
from accelerate import Accelerator

import torch
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR

import xformers

from transformers import CLIPTextModel, CLIPTokenizer, AutoModel, AutoTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from src.stable_diffusion.dataset import MelonDataset, valid_prompt
from src.stable_diffusion.train import train
from src.stable_diffusion.utils.util import set_seed


def main(args):
    # 1. Set default value-1
    if args.use_wandb:
        wandb.init(
            name=args.exp_name + "_Resume" if args.resume else args.exp_name,
            project="gen_album_cover",
            entity="ganisokay",
            config=args,
        )

    set_seed(args.seed)

    args.output_dir = os.path.join(
        args.base_path, args.exp_path, args.exp_name, args.output_dir
    )
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision

    # 2. Load model
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler"
    )

    if args.text_model == "clip":
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model, subfolder="text_encoder", revision=args.revision
        )
    elif args.text_model == "klue":
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base", use_fast=False)
        text_encoder = AutoModel.from_pretrained("klue/roberta-base")

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, subfolder="vae", revision=args.revision
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model, subfolder="unet", revision=args.non_ema_revision
    )

    vae.requires_grad_(False)  # Freeze
    text_encoder.requires_grad_(False)  # Freeze

    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model, subfolder="unet"
        )
        ema_unet = EMAModel(ema_unet)
    else:
        ema_unet = None

    if args.use_xformers:
        unet.enable_xformers_memory_efficient_attention()

    # 3. Prepare Dataset
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    args.csv_path = os.path.join(
        args.base_path, args.exp_path, args.exp_name, args.csv_path
    )
    train_dataset = MelonDataset(args.csv_path, train_transforms, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    args.txt_path = os.path.join(
        args.base_path, args.exp_path, args.exp_name, args.txt_path
    )
    args.valid_prompts = valid_prompt(args.txt_path, args.max_prompt)

    # 4. Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 5. Learning rate scheduler Using diffusers.get_scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.grad_accum,
        num_training_steps=args.max_train_steps * args.grad_accum,
    )

    # 6. Prepare accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    def save_model_hook(models, weights, output_dir):
        if args.use_ema:
            ema_unet.averaged_model.save_pretrained(
                os.path.join(output_dir, "unet_ema")
            )

        for i, model in enumerate(models):
            model.save_pretrained(os.path.join(output_dir, "unet"))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(models, input_dir):
        if args.use_ema:
            load_model = UNet2DConditionModel.from_pretrained(
                os.path.join(input_dir, "unet_ema")
            )
            ema_unet = EMAModel(load_model)
            ema_unet.averaged_model.to(accelerator.device)
            del load_model

        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(
                input_dir, subfolder="unet"
            )
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.use_ema:
        ema_unet.averaged_model.to(accelerator.device)

    text_encoder.to(accelerator.device, dtype=weight_dtype)  # Cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)  # Cast to weight_dtype

    # 7. Set default value-2
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum)
    args.max_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    first_epoch = 0
    resume_step = 0

    if args.resume:
        dirs = os.listdir(args.output_dir)  # Get the most recent checkpoint
        dirs = [dir for dir in dirs if dir.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            print(f"Checkpoint '{args.resume}' does not exist")
            return
        else:
            print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.grad_accum
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.grad_accum
            )

    # 8. Training
    train(
        args,
        accelerator,
        weight_dtype,
        noise_scheduler,
        tokenizer,
        text_encoder,
        vae,
        unet,
        ema_unet,
        train_dataloader,
        optimizer,
        lr_scheduler,
        global_step,
        first_epoch,
        resume_step,
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    # 1. Set default value-1
    parser.add_argument("--use-wandb", type=bool, default=False)
    parser.add_argument("--exp-name", type=str, default="[tag]ExpName")

    parser.add_argument("--seed", type=int, default=3)

    parser.add_argument(
        "--base-path",
        type=str,
        default="/opt/ml/input/code/level3_cv_finalproject-cv-03/stable_diffusion",
        help="Set base path",
    )
    parser.add_argument(
        "--exp-path", type=str, default="experiments", help="Set base experiment path"
    )
    parser.add_argument("--output-dir", type=str, default="checkpoints")

    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--mixed-precision", type=str, default="fp16")

    # 2. Load model
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="https://huggingface.co/CompVis",
    )
    parser.add_argument("--text-model", type=str, default="clip", help="[clip, klue]")

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--non-ema-revision",
        type=str,
        default=None,
        help="Revision of pretrained non-ema model identifier",
    )

    parser.add_argument(
        "--use-ema",
        type=bool,
        default=False,
        help="EMA(Exponential Moving Average) : Better results and more reliable training",
    )
    parser.add_argument(
        "--use-xformers",
        type=bool,
        default=False,
        help="https://facebookresearch.github.io/xformers/what_is_xformers.html",
    )

    # 3. Prepare Dataset
    parser.add_argument("--resolution", nargs="+", type=int, default=[256, 256])

    parser.add_argument("--csv-path", type=str, default="albums.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--txt-path", type=str, default="prompts.txt")
    parser.add_argument(
        "--max-prompt",
        type=int,
        default=16,
        help="Up to 16 because it prevents memory problems and performs Wandb",
    )

    # 4. Optimizer(Adam)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-2)
    parser.add_argument("--adam-epsilon", type=float, default=1e-08)

    # 5. Learning rate scheduler Using diffusers.get_scheduler
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="constant",
        help=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=500)

    # 6. Prepare accelerator

    # 7. Set default value-2
    parser.add_argument(
        "--max-train-steps", type=int, default=100_000, help="Override max-train-steps"
    )
    parser.add_argument("--resume", type=bool, default=False)

    # 8. Training
    parser.add_argument(
        "--noise-offset", type=float, default=0, help="Recommended value is 0.1"
    )
    parser.add_argument(
        "--input-perturbation", type=float, default=0, help="Recommended value is 0.1"
    )
    parser.add_argument(
        "--snr-gamma", type=float, default=None, help="Recommended value is 5.0"
    )

    parser.add_argument("--ckpt-step", type=int, default=500, help="Save a checkpoint")
    parser.add_argument(
        "--ckpt-max", type=int, default=2, help="Max number of checkpoints to store"
    )
    parser.add_argument("--valid-epoch", type=int, default=5, help="Run validation")

    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument(
        "--save-img-path",
        type=str,
        default="valid",
        help="results path | if save-img-path is None, don't save grid",
    )
    parser.add_argument(
        "--grid-size",
        nargs="+",
        type=int,
        default=[4, 4],
        help="!python main.py --grid-size 1 2 >> args.grid_size=[1, 2]",
    )

    args = parser.parse_args()

    main(args)
