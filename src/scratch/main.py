# Python built-in modules
import os
import io
import base64
import uuid
from datetime import datetime

# Pytorch
import torch
from torch import cuda

# Backend
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse

from pydantic import BaseModel

# Other modules
import numpy as np
from pytz import timezone

# Built-in modules
from .gpt3_api import get_description
from .gcp.bigquery import BigQueryLogger
from .gcp.cloud_storage import GCSUploader
from .gcp.error import ErrorReporter
from .model import AlbumModel
from .utils import load_yaml


# Load config
gcp_config = load_yaml(os.path.join("src/scratch/config", "private.yaml"), "gcp")
public_config = load_yaml(os.path.join("src/scratch/config", "public.yaml"))

# Start fastapi
app = FastAPI()

bigquery_logger = BigQueryLogger(gcp_config)
gcs_uploader = GCSUploader(gcp_config)
error_reporter = ErrorReporter(gcp_config)

device = "cuda" if cuda.is_available() else "cpu"

# Generate a unique ID for this request
request_id = str(uuid.uuid4())


def load_model():
    model = AlbumModel(public_config["model"], public_config["language"], device)
    return model


# Album input Schema
class AlbumInput(BaseModel):
    song_names: str
    artist_name: str
    genre: str
    album_name: str
    lyric: str


# Review input Schema
class ReviewInput(BaseModel):
    rating: int
    comment: str


# REST API - Post ~/generate_cover
@app.post("/generate_cover")
async def generate_cover(album: AlbumInput, model: AlbumModel = Depends(load_model)):
    images = []
    urls = []

    summarization = get_description(
        album.lyric,
        album.artist_name,
        album.album_name,
        album.song_names,
    )

    seeds = np.random.randint(
        public_config["generate"]["max_seed"], size=public_config["generate"]["n_gen"]
    )

    for i, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(int(seed))

        # Generate Images
        with torch.no_grad():
            image = model.pipeline(
                prompt=f"A photo of a {album.genre} album cover with a {summarization} atmosphere visualized.",
                num_inference_steps=public_config["generate"]["inference_step"],
                generator=generator,
            ).images[0]

        image = image.resize(
            (public_config["generate"]["height"], public_config["generate"]["width"])
        )

        # Convert to base64-encoded string
        byte_arr = io.BytesIO()
        image.save(byte_arr, format=public_config["generate"]["save_format"])
        byte_arr = byte_arr.getvalue()
        base64_str = base64.b64encode(byte_arr).decode()

        urls.append(
            [
                byte_arr,
                f"{request_id}_image_{i}.{public_config['generate']['save_format']}",
            ]
        )

        images.append(base64_str)

    # Upload to GCS
    image_urls = gcs_uploader.save_image_to_gcs(urls)

    # Log to BigQuery
    album_log = {
        "request_id": request_id,
        "request_time": datetime.utcnow()
        .astimezone(timezone("Asia/Seoul"))
        .isoformat(),
        "song_names": album.song_names,
        "artist_name": album.artist_name,
        "genre": album.genre,
        "album_name": album.album_name,
        "lyric": album.lyric,
        "summarization": summarization,
        "image_urls": image_urls,
        "language": public_config["language"],
    }
    bigquery_logger.log(album_log, "user_album")

    return {"images": images}


# REST API - Post ~/review
@app.post("/review")
async def review(review: ReviewInput):
    # Log to BigQuery
    review_log = {
        "request_id": request_id,
        "request_time": datetime.utcnow()
        .astimezone(timezone("Asia/Seoul"))
        .isoformat(),
        "rating": review.rating,
        "comment": review.comment,
        "language": public_config["language"],
    }

    bigquery_logger.log(review_log, "user_review")

    return review


# Exception handling using google cloud
@app.exception_handler(Exception)
async def handle_exceptions(request: Request, exc: Exception):
    error_reporter.python_error()

    return JSONResponse(status_code=500, content={"message": "Internal Server Error"})
