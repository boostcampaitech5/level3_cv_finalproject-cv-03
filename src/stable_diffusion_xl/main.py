import os
import math
from argparse import ArgumentParser
import itertools
from typing import Dict

import numpy as np
import wandb
from accelerate import Accelerator

import torch
from torchvision import transforms
import torch.nn.functional as F

from transformers import CLIPTextModel, AutoTokenizer, CLIPTextModelWithProjection

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnProcessor2_0
from diffusers.optimization import get_scheduler

from .dataset import MelonDatasetSDXL, valid_prompt, collate_fn
from .train import training
from .utils.util import set_seed
from .utils.training import compute_additional_embeddings


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

    # Load the tokenizers & text encoder
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model,
        subfolder="text_encoder",
        revision=args.revision,
    )

    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model,
        subfolder="text_encoder_2",
        revision=args.revision,
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, subfolder="vae", revision=args.revision
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model, subfolder="unet", revision=args.revision
    )

    vae.requires_grad_(False)  # Freeze
    text_encoder_one.requires_grad_(False)  # Freeze
    text_encoder_two.requires_grad_(False)  # Freeze
    unet.requires_grad_(False)  # Freeze

    vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    unet_lora_attn_procs = {}
    unet_lora_parameters = []
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_processor_class = (
            LoRAAttnProcessor2_0
            if hasattr(F, "scaled_dot_product_attention")
            else LoRAAttnProcessor
        )
        module = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        )
        unet_lora_attn_procs[name] = module
        unet_lora_parameters.extend(module.parameters())

    unet.set_attn_processor(unet_lora_attn_procs)

    text_lora_parameters_one = LoraLoaderMixin._modify_text_encoder(
        text_encoder_one, dtype=torch.float32
    )
    text_lora_parameters_two = LoraLoaderMixin._modify_text_encoder(
        text_encoder_two, dtype=torch.float32
    )

    model = StableDiffusionXLPipeline(
        vae,
        text_encoder_one,
        text_encoder_two,
        tokenizer_one,
        tokenizer_two,
        unet,
        noise_scheduler,
    )

    # 3. Prepare Dataset

    unet_added_cond_kwargs = compute_additional_embeddings(
        args.resolution[0], args.resolution[1], weight_dtype
    )

    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    args.csv_path = os.path.join(
        args.base_path, args.exp_path, args.exp_name, args.csv_path
    )
    train_dataset = MelonDatasetSDXL(
        args.csv_path,
        train_transforms,
        [tokenizer_one, tokenizer_two],
        unet_added_cond_kwargs,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    args.txt_path = os.path.join(
        args.base_path, args.exp_path, args.exp_name, args.txt_path
    )
    args.valid_prompts = valid_prompt(args.txt_path, args.max_prompt)

    # 4. Optimizer
    params_to_optimize = itertools.chain(
        unet_lora_parameters, text_lora_parameters_one, text_lora_parameters_two
    )
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 5. Learning rate scheduler Using diffusers.get_scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 6. Prepare accelerator
    (
        model.unet,
        model.text_encoder,
        model.text_encoder_2,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model.unet,
        model.text_encoder,
        model.text_encoder_2,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    def unet_attn_processors_state_dict(unet) -> Dict[str, torch.tensor]:
        """
        Returns:
            a state dict containing just the attention processor parameters.
        """
        attn_processors = unet.attn_processors

        attn_processors_state_dict = {}

        for attn_processor_key, attn_processor in attn_processors.items():
            for parameter_key, parameter in attn_processor.state_dict().items():
                attn_processors_state_dict[
                    f"{attn_processor_key}.{parameter_key}"
                ] = parameter

        return attn_processors_state_dict

    def save_model_hook(models, weights, output_dir):
        # there are only two options here. Either are just the unet attn processor layers
        # or there are the unet and text encoder atten layers
        unet_lora_layers_to_save = None
        text_encoder_one_lora_layers_to_save = None
        text_encoder_two_lora_layers_to_save = None

        for m in models:
            if isinstance(m, type(accelerator.unwrap_model(unet))):
                unet_lora_layers_to_save = unet_attn_processors_state_dict(m)
            elif isinstance(m, type(accelerator.unwrap_model(text_encoder_one))):
                text_encoder_one_lora_layers_to_save = text_encoder_lora_state_dict(m)
            elif isinstance(m, type(accelerator.unwrap_model(text_encoder_two))):
                text_encoder_two_lora_layers_to_save = text_encoder_lora_state_dict(m)
            else:
                raise ValueError(f"unexpected save model: {m.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

        StableDiffusionXLPipeline.save_lora_weights(
            output_dir,
            unet_lora_layers=unet_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
        )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            m = models.pop()

            if isinstance(m, type(accelerator.unwrap_model(unet))):
                unet_ = m
            elif isinstance(m, type(accelerator.unwrap_model(text_encoder_one))):
                text_encoder_one_ = m
            elif isinstance(m, type(accelerator.unwrap_model(text_encoder_two))):
                text_encoder_two_ = m
            else:
                raise ValueError(f"unexpected save model: {m.__class__}")

        lora_state_dict, network_alpha = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(
            lora_state_dict, network_alpha=network_alpha, unet=unet_
        )
        LoraLoaderMixin.load_lora_into_text_encoder(
            lora_state_dict, network_alpha=network_alpha, text_encoder=text_encoder_one_
        )
        LoraLoaderMixin.load_lora_into_text_encoder(
            lora_state_dict, network_alpha=network_alpha, text_encoder=text_encoder_two_
        )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

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
    training(
        args,
        accelerator,
        weight_dtype,
        model,
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
        default="/opt/ml/input/code/level3_cv_finalproject-cv-03/src/stable_diffusion_xl",
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
        default="stabilityai/stable-diffusion-xl-base-0.9",
        help="https://huggingface.co/CompVis",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    # 3. Prepare Dataset
    parser.add_argument("--resolution", nargs="+", type=int, default=[1024, 1024])

    parser.add_argument("--csv-path", type=str, default="albums.csv")
    parser.add_argument("--batch-size", type=int, default=8)
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
    parser.add_argument("--ckpt-step", type=int, default=500, help="Save a checkpoint")
    parser.add_argument(
        "--ckpt-max", type=int, default=2, help="Max number of checkpoints to store"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Clip gradient"
    )
    parser.add_argument("--valid-epoch", type=int, default=5, help="Run validation")

    parser.add_argument("--num-inference-steps", type=int, default=100)
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
