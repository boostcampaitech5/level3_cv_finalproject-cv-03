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

from gpt3_api import get_description

from datetime import datetime
import uuid

from bigquery_logger import BigQueryLogger
from cloud_storage_manager import GCSUploader
from model import Model

app = FastAPI()

model = Model()
bigquery_logger = BigQueryLogger()
gcs_uploader = GCSUploader()

# Generate a unique ID for this request
request_id = str(uuid.uuid4())


# Input Schema
class AlbumInput(BaseModel):
    song_names: str
    artist_name: str
    genre: str
    album_name: str
    release: str
    lyric: str


class ReviewInput(BaseModel):
    rating: int
    comment: str


# Load model on Start (Will be changed)
@app.on_event("startup")
def load_model():
    model.load()


@app.post("/generate_cover")
async def generate_cover(album: AlbumInput):
    device = "cuda" if cuda.is_available() else "cpu"
    images = []
    image_urls = []

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

    for i in range(4):
        seed = random.randint(100)
        generator = torch.Generator(device=device).manual_seed(seed)

        # Generate Images
        with torch.no_grad():
            image_tensor = model.pipeline(
                prompt=f"A photo of a {album.genre} album cover with a {summarization} atmosphere visualized.",
                num_inference_steps=20,
                generator=generator,
            ).images[0]

        image = image_tensor
        image = image.resize((256, 256))

        # Convert to base64-encoded string
        byte_arr = io.BytesIO()
        image.save(byte_arr, format="JPEG")
        byte_arr = byte_arr.getvalue()
        base64_str = base64.b64encode(byte_arr).decode()

        # Upload to GCS
        image_url = gcs_uploader.save_image_to_gcs(image, f"{request_id}_image_{i}.jpg")
        image_urls.append(image_url)

        images.append(base64_str)

    # Log to BigQuery
    bigquery_logger.log(album, summarization, request_id, image_urls)

    return {"images": images}


@app.post("/review")
async def review(review: ReviewInput):
    bigquery_logger.log_review(review, request_id)
    return review
