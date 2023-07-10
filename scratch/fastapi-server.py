from fastapi import FastAPI, Depends, Response
from pydantic import BaseModel
from typing import List

import pandas as pd
from numpy import random
from PIL import Image
import io
import base64

import torch
from torch import cuda

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

import openai
from gpt3_api import get_description


app = FastAPI()

pipeline = None


# Input Schema
class AlbumInput(BaseModel):
    song_names: str
    artist_name: str
    genre: str
    album_name: str
    release: str
    lyric: str


# 시작시 model load
# pipeline을 global variable로 설정했지만 변경 예정
@app.on_event("startup")
def load_model():
    global pipeline
    device = "cuda" if cuda.is_available() else "cpu"
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipeline = pipeline.to(device)
    pipeline.enable_xformers_memory_efficient_attention()


# pipeline이 global variable이라서 그냥 사용하지만, 보통은 의존성 주입(Depends) 같은 방법으로 사용하는 듯 합니다

# 예시) async def generate_cover(album: AlbumInput, pipeline: StableDiffusionPipeline = Depends(get_pipeline)):
"""def get_pipeline():
    return pipeline"""


@app.post("/generate_cover")
async def generate_cover(album: AlbumInput):
    device = "cuda" if cuda.is_available() else "cpu"
    images = []

    # Determine season from release date
    month = int(album.release.split("-")[1])
    if month > 11 or month < 3:
        season = "winter"
    elif 3 <= month < 6:
        season = "spring"
    elif 6 <= month < 9:
        season = "summer"
    elif 9 <= month < 12:
        season = "fall"

    summarization = get_description(
        album.lyric, album.artist_name, album.album_name, season, album.song_names
    )

    for _ in range(4):
        seed = random.randint(100)
        generator = torch.Generator(device=device).manual_seed(seed)

        with torch.no_grad():
            image_tensor = pipeline(
                prompt=f"A photo of a {album.genre} album cover with a {summarization} atmosphere visualized.",
                num_inference_steps=20,
                generator=generator,
            ).images[0]

        image = image_tensor
        image = image.resize((256, 256))

        # base64-encoded string으로 변환
        byte_arr = io.BytesIO()
        image.save(byte_arr, format="JPEG")
        byte_arr = byte_arr.getvalue()
        base64_str = base64.b64encode(byte_arr).decode()

        images.append(base64_str)

    return {"images": images}
