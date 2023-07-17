from argparse import ArgumentParser
import os

import torch
from torch import cuda

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from utils.util import set_seed


def test(args):
    # 1. Set seed
    set_seed(args.seed)

    # 2. Set path
    base_path = os.path.join(args.base_path, args.exp_path, args.exp_name)

    # 3. Load model
    unet = UNet2DConditionModel.from_pretrained(
        os.path.join(base_path, "checkpoints", args.ckpt), subfolder="unet"
    )

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        unet=unet,
    )

    pipeline = pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=True)

    # https://facebookresearch.github.io/xformers/what_is_xformers.html
    if args.use_xformers:
        pipeline.enable_xformers_memory_efficient_attention()

    # 4. Generate image
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    with torch.autocast(args.device):
        image = pipeline(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[
            0
        ]  # Default: 50

    # 5. Save
    image.save(os.path.join(base_path, "results", "result.png"))


if __name__ == "__main__":
    parser = ArgumentParser()

    # 1. Set seed
    parser.add_argument("--seed", type=int, default=3)

    # 2. Set path
    parser.add_argument(
        "--base-path",
        type=str,
        default="/opt/ml/input/code/level3_cv_finalproject-cv-03/stable_diffusion",
        help="Set base path",
    )
    parser.add_argument(
        "--exp-path", type=str, default="experiments", help="Set base experiment path"
    )
    parser.add_argument("--exp-name", type=str, default="[tag]ExpName")
    parser.add_argument("--ckpt", type=str, default="checkpoint-3")

    # 3. Load model
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="https://huggingface.co/CompVis",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if cuda.is_available() else "cpu"
    )
    parser.add_argument("--use-xformers", type=bool, default=False)

    # 4. Generate image
    parser.add_argument("--prompt", type=str, default="A photo of album cover")
    parser.add_argument("--num-inference-steps", type=int, default=20)

    args = parser.parse_args()

    args.prompt = input("Prompt >> ")

    test(args)
