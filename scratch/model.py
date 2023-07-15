# huggingface - transformers
from transformers import CLIPTextModel, CLIPTokenizer, AutoModel, AutoTokenizer

# huggingface - diffusers
from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel


class AlbumModel:
    def __init__(self, model_config: dict, lang: str, device: str):
        self.device = device
        self.model_config = model_config
        if lang == "EN":
            self.text_encoder = (
                CLIPTextModel.from_pretrained(
                    "CompVis/stable-diffusion-v1-4", subfolder="text_encoder"
                ),
            )
            self.tokenizer = (
                CLIPTokenizer.from_pretrained(
                    "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
                ),
            )
        elif lang == "KR":
            self.text_encoder = AutoModel.from_pretrained("klue/roberta-base")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "klue/roberta-base", use_fast=False
            )

    def get_model(self) -> None:
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_config["stable_diffusion"],
            unet=UNet2DConditionModel.from_pretrained(
                self.model_config["unet"], subfolder="unet"
            ),
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
        )
        self.pipeline = self.pipeline.to(self.device)
        if self.model_config["xformers"]:
            self.pipeline.enable_xformers_memory_efficient_attention()
