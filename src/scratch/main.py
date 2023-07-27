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
from fastapi import FastAPI, Request, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
from pydantic import BaseModel

# Other modules
import numpy as np
from pytz import timezone
from PIL import Image
import subprocess
import shlex
import random
import string
from huggingface_hub import login

# Built-in modules
from .gpt3_api import get_description, get_dreambooth_prompt
from .gcp.bigquery import BigQueryLogger
from .gcp.cloud_storage import GCSUploader
from .gcp.error import ErrorReporter
from .model import AlbumModel
from .utils import load_yaml
import urllib3
from urllib3.exceptions import InsecureRequestWarning

urllib3.disable_warnings(InsecureRequestWarning)

# Load config
gcp_config = load_yaml(
    os.path.join(
        "/opt/ml/level3_cv_finalproject-cv-03/src/scratch/config", "private.yaml"
    ),
    "gcp",
)
huggingface_config = load_yaml(
    os.path.join(
        "/opt/ml/level3_cv_finalproject-cv-03/src/scratch/config", "private.yaml"
    ),
    "huggingface",
)
public_config = load_yaml(
    os.path.join(
        "/opt/ml/level3_cv_finalproject-cv-03/src/scratch/config", "public.yaml"
    )
)
train_config = load_yaml(
    os.path.join(
        "/opt/ml/level3_cv_finalproject-cv-03/src/scratch/config", "dreambooth.yaml"
    )
)

# Start fastapi
app = FastAPI()

login(token=huggingface_config["token"], add_to_git_credential=True)

bigquery_logger = BigQueryLogger(gcp_config)
gcs_uploader = GCSUploader(gcp_config)
error_reporter = ErrorReporter(gcp_config)

device = "cuda" if cuda.is_available() else "cpu"


def load_model():
    model = AlbumModel(public_config["model"], public_config["language"], device)
    return model


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


token = get_random_string(5)

# @app.on_event("startup")
# async def startup_event():
#     global model
#     model = load_model()
#     print("Model loaded successfully!")


# Album input Schema
class AlbumInput(BaseModel):
    song_names: str
    artist_name: str
    genre: str
    album_name: str
    lyric: str


class UserInput(BaseModel):
    gender: str


# Review input Schema
class ReviewInput(BaseModel):
    rating: int
    comment: str


class ImageInput(BaseModel):
    file_path: str  # 이미지 url(upload를 통해 gcs에 들어가면 생기는 url)
    width: int  # 이미지 너비
    height: int  # 이미지 높이
    format: str  # 이미지 포맷 (e.g., JPEG, PNG )


# REST API - Post ~/generate_cover
@app.post("/generate_cover")
async def generate_cover(album: AlbumInput):
    global request_id
    request_id = str(uuid.uuid4())

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


@app.post("/upload_image")
async def upload_image(image: UploadFile = File(...)):
    global request_id
    request_id = str(uuid.uuid4())

    # Read the image file
    image_bytes = await image.read()

    # Generate a unique name for the image based on the request_id and original filename
    destination_filename = f"{image.filename}"

    # Define the directory where to save the image
    image_dir = (
        Path("/opt/ml/level3_cv_finalproject-cv-03/dreambooth/data/users") / token
    )

    # Create the directory if it does not exist
    image_dir.mkdir(parents=True, exist_ok=True)

    # Save the image locally
    with open(image_dir / destination_filename, "wb") as buffer:
        buffer.write(image_bytes)

    return {"image_url": str(image_dir / destination_filename)}


@app.post("/train")
async def train(user: UserInput):
    seeds = np.random.randint(100000)
    # Run the train.py script as a separate process
    process = subprocess.Popen(
        [
            "python",
            "/opt/ml/level3_cv_finalproject-cv-03/dreambooth/run.py",
            "--config-file",
            "/opt/ml/level3_cv_finalproject-cv-03/src/scratch/config/dreambooth.yaml",
            "--token",
            token,
            "--user-gender",
            user.gender,
            "--seed",
            str(seeds),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout, stderr = process.communicate()

    if process.returncode != 0:
        return {"status": "error", "message": stderr.decode()}
    command = stdout.decode()

    subprocess.run(command, shell=True)

    return {"status": "Train started", "message": command}


@app.post("/inference")
async def inference(album: AlbumInput, user: UserInput):
    class_name = user.gender
    summarization = get_dreambooth_prompt(
        album.lyric,
        album.album_name,
        album.song_names,
        class_name,
        album.genre,
        album.artist_name,
    )
    # Run the train.py script as a separate process
    process = subprocess.Popen(
        [
            "python",
            "/opt/ml/level3_cv_finalproject-cv-03/dreambooth/inference.py",
            "--user-id",
            token,
            "--prompt",
            summarization,
            "--user-gender",
            class_name,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Get the output and error messages from the process
    stdout, stderr = process.communicate()

    # Check if the process completed successfully
    if process.returncode != 0:
        return {"status": "error", "message": stderr.decode()}

    saved_dir = f"/opt/ml/level3_cv_finalproject-cv-03/dreambooth/data/results/{token}"

    files = os.listdir(saved_dir)
    urls = []

    for file in files:
        # Only process image files (png, jpg, etc.)
        if file.endswith((".png", ".jpg", ".jpeg")):
            # Open image file in binary mode and read it
            with open(os.path.join(saved_dir, file), "rb") as image_file:
                byte_arr = image_file.read()

            blob_name = f"{token}/{file}"
            # Append the tuple (byte_arr, desired_blob_name) to urls
            urls.append((byte_arr, blob_name))

    # Upload the images to GCS and get their URLs
    image_urls = gcs_uploader.save_image_to_gcs(
        urls, bucket_name="generated-dreambooth-images"
    )
    prompt = stdout.decode()

    return {"status": "your images are created and saved!", "prompt": prompt}
