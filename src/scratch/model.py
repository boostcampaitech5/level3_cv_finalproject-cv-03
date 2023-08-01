# Python built-in modules
import os

# torch
import torch

# huggingface - transformers
from transformers import CLIPTextModel, CLIPTokenizer, AutoModel, AutoTokenizer

# huggingface - diffusers
from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)

# User Defined Modules
from .utils import load_yaml

private_config = load_yaml(os.path.join("src/scratch/config", "private.yaml"))


class StableDiffusion:
    def __init__(self, model_config: dict, lang: str, device: str):
        self.device = device
        self.model_config = model_config
        if lang == "EN":
            self.text_encoder = CLIPTextModel.from_pretrained(
                "CompVis/stable-diffusion-v1-4", subfolder="text_encoder"
            )
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
            )
        elif lang == "KR":
            self.text_encoder = AutoModel.from_pretrained("klue/roberta-base")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "klue/roberta-base", use_fast=False
            )

        self.pipeline = self.get_model()

    def get_model(self) -> None:
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_config["SD"]["stable_diffusion"],
            unet=UNet2DConditionModel.from_pretrained(
                self.model_config["SD"]["unet"], subfolder="unet"
            ),
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        )
        pipeline = pipeline.to(self.device)
        if self.model_config["SD"]["xformers"]:
            pipeline.enable_xformers_memory_efficient_attention()

        return pipeline


## -- 아래에 Stable Diffusion XL 모델 추가해주세요
class StableDiffusionXL:
    def __init__(self, model_config: dict, device: str):
        self.device = device
        self.model_config = model_config
        self.pipeline = self.get_model()

    def get_model(self) -> None:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_config["SDXL"]["stable_diffusion"],
            torch_dtype=torch.float16,
            variant="fp16",
            token=private_config["huggingface"]["sdxl_token"],
        ).to("cuda")

        pipe.load_lora_weights(self.model_config["SDXL"]["ckpt"], use_safetensors=False)

        return pipe
