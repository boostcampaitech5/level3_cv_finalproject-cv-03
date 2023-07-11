import torch
from torch import cuda

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler


class Model:
    def __init__(self):
        self.pipeline = None

    def load(self):
        device = "cuda" if cuda.is_available() else "cpu"
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4"
        )
        self.pipeline = self.pipeline.to(device)
        self.pipeline.enable_xformers_memory_efficient_attention()
