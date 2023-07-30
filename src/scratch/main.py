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
from fastapi import FastAPI, Request, Depends, APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

# Other modules
import numpy as np
from pytz import timezone
from typing import List, Dict
from pathlib import Path
import subprocess
import random
import string
import urllib3
from typing import Optional
from typing import List

# Built-in modules
from .gpt3_api import get_description, get_dreambooth_prompt
from .gcp.bigquery import BigQueryLogger
from .gcp.cloud_storage import GCSUploader
from .gcp.error import ErrorReporter
from .model import AlbumModel
from .utils import load_yaml

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load config
gcp_config = load_yaml(os.path.join("src/scratch/config", "private.yaml"), "gcp")
public_config = load_yaml(os.path.join("src/scratch/config", "public.yaml"))
train_config = load_yaml(os.path.join("src/scratch/dreambooth", "dreambooth.yaml"))
bigquery_config = gcp_config["bigquery"]

# Start fastapi
app = FastAPI()
api_router = APIRouter(prefix="/api")

# --- 정리 예정, Refactoring x, Configuration x ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------

bigquery_logger = BigQueryLogger(gcp_config)
gcs_uploader = GCSUploader(gcp_config)
error_reporter = ErrorReporter(gcp_config)

device = "cuda" if cuda.is_available() else "cpu"


def load_model():
    model = AlbumModel(public_config["model"], public_config["language"], device)
    return model


@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()
    print("Model loaded successfully!")


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


token = get_random_string(5)


# Schema for get_album_images
class AlbumImage(BaseModel):
    song_name: str
    artist_name: str
    album_name: str
    genre: str
    lyric: str
    gender: str
    create_date: str
    url: str


class UserInfo(BaseModel):
    nickname: str
    age_range: str
    email: str


class UserAlbumInput(BaseModel):
    user_id: str
    model: str
    song_name: str
    artist_name: str
    album_name: str
    genre: str
    lyric: str
    gender: str
    image_urls: List[str]


class UserReviewInput(BaseModel):
    output_id: str
    url_id: int
    user_id: str
    rating: int
    comment: str


# 개인정보 보호를 위해 이메일 도메인주소 마스킹하기
def mask_email_domain(email):
    username, domain = email.split("@", 1)
    masked_domain = domain[:2] + "*" * (len(domain) - 2)
    masked_email = f"{username}@{masked_domain}"
    return masked_email


@api_router.post("/user")
async def user_login(userinfo: UserInfo):
    # Log to BigQuery

    dataset_id = gcp_config["bigquery"]["dataset_id"]
    user_table = gcp_config["bigquery"]["table_id"]["user"]

    masked_email = mask_email_domain(userinfo.email)
    query = f"""
           SELECT user_id FROM `{dataset_id}.{user_table}`
           WHERE email = '{masked_email}';
        """
    query_job = bigquery_logger.client.query(query)
    result = list(query_job)

    if len(result) == 0:  # 새로운 유저 정보 저장
        print("New User Login!")
        user_id = str(uuid.uuid4())
        create_date = datetime.utcnow().astimezone(timezone("Asia/Seoul")).isoformat()
        User = {
            "user_id": user_id,
            "nickname": userinfo.nickname,
            "age_range": userinfo.age_range,
            "email": masked_email,
            "create_date": create_date,
        }
        bigquery_logger.log(User, "user")
        LoginLog = {
            "login_log_id": str(uuid.uuid4()),
            "user_id": user_id,
            "login_date": create_date,
        }
        bigquery_logger.log(LoginLog, "login_log")
        return {"user_id": user_id}
    else:  # 기존 유저는 로그인 로그만 저장
        print("Existing User Login!")
        LoginLog = {
            "login_log_id": str(uuid.uuid4()),
            "user_id": result[0]["user_id"],
            "login_date": datetime.utcnow()
            .astimezone(timezone("Asia/Seoul"))
            .isoformat(),
        }
        bigquery_logger.log(LoginLog, "login_log")
        return {"user_id": result[0]["user_id"]}


@api_router.post("/generate_cover")
async def generate_cover(input: UserAlbumInput):
    global request_id
    request_id = str(uuid.uuid4())

    input_log = {
        "input_id": request_id,
        "user_id": input.user_id,
        "model": input.model,
        "song_name": input.song_name,
        "artist_name": input.artist_name,
        "album_name": input.album_name,
        "genre": input.genre,
        "lyric": input.lyric,
        "gender": input.gender,
        "image_urls": input.image_urls,
        "create_date": datetime.utcnow().astimezone(timezone("Asia/Seoul")).isoformat(),
    }
    bigquery_logger.log(input_log, "input")

    images = []
    urls = []

    summarization = get_description(
        input.lyric,
        input.artist_name,
        input.album_name,
        input.song_name,
    )

    seeds = np.random.randint(
        public_config["generate"]["max_seed"], size=public_config["generate"]["n_gen"]
    )
    prompt = f"A photo of a {input.genre} album cover with a {summarization} atmosphere visualized."
    for i, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(int(seed))

        # Generate Images
        with torch.no_grad():
            image = model.pipeline(
                prompt=prompt,
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
        # base64_str = base64.b64encode(byte_arr).decode()

        urls.append(
            [
                byte_arr,
                f"{request_id}_image_{i}.{public_config['generate']['save_format']}",
            ]
        )

        images.append(byte_arr)

    # Upload to GCS
    image_urls = gcs_uploader.save_image_to_gcs(urls)

    # Log to BigQuery
    output_id = str(uuid.uuid4())
    output_log = {
        "output_id": output_id,
        "input_id": request_id,
        "image_urls": image_urls,
        "seeds": [int(seed) for seed in seeds],
        "prompt": prompt,
        "create_date": datetime.utcnow().astimezone(timezone("Asia/Seoul")).isoformat(),
    }
    bigquery_logger.log(output_log, "output")

    return {"images": image_urls, "output_id": output_id}


# REST API - Post ~/review
# @app.post("/review")
@api_router.post("/review")
async def review(review: UserReviewInput):
    # Log to BigQuery
    review_id = str(uuid.uuid4())
    review_log = {
        "review_id": review_id,
        "output_id": review.output_id,
        "url_id": review.url_id,
        "user_id": review.user_id,
        "rating": review.rating,
        "comment": review.comment,
        "create_date": datetime.utcnow().astimezone(timezone("Asia/Seoul")).isoformat(),
    }

    bigquery_logger.log(review_log, "review")
    return review


@api_router.get("/get_album_images", response_model=Dict[str, List[AlbumImage]])
# @app.get("/get_album_images", response_model=Dict[str, List[AlbumImage]])
async def get_album_images(user: Optional[str] = None):
    # Query to retrieve latest and best-rated images from BigQuery

    dataset_id = gcp_config["bigquery"]["dataset_id"]
    review_table = gcp_config["bigquery"]["table_id"]["review"]
    output_table = gcp_config["bigquery"]["table_id"]["output"]
    input_table = gcp_config["bigquery"]["table_id"]["input"]

    if user:
        query = f"""
            SELECT input.song_name, input.artist_name, input.album_name, input.genre, input.lyric,
            input.gender, review.create_date, output.image_urls[review.url_id-1] AS image_url
            FROM {dataset_id}.{review_table} AS review
            JOIN {dataset_id}.{output_table} AS output
            ON review.output_id = output.output_id
            JOIN {dataset_id}.{input_table} AS input
            ON output.input_id = input.input_id
            WHERE review.user_id = '{user}'
            ORDER BY review.create_date DESC
            LIMIT 20
        """
    else:
        query = f"""
            SELECT input.song_name, input.artist_name, input.album_name, input.genre, input.lyric,
            input.gender, review.create_date, output.image_urls[review.url_id-1] AS image_url
            FROM {dataset_id}.{review_table} AS review
            JOIN {dataset_id}.{output_table} AS output
            ON review.output_id = output.output_id
            JOIN {dataset_id}.{input_table} AS input
            ON output.input_id = input.input_id
            ORDER BY review.rating DESC, review.create_date DESC
            LIMIT 12
        """

    # Execute the query
    query_job = bigquery_logger.client.query(query)
    results = query_job.result()

    # Process the results and create a list of AlbumImage objects
    album_images = []
    for row in results:
        album_image = AlbumImage(
            song_name=row["song_name"],
            artist_name=row["artist_name"],
            album_name=row["album_name"],
            genre=row["genre"],
            lyric=row["lyric"],
            gender=row["gender"],
            create_date=str(row["create_date"]),
            url=row["image_url"],
        )
        album_images.append(album_image)

    return {"album_images": album_images}


# Exception handling using google cloud
@app.exception_handler(Exception)
async def handle_exceptions(request: Request, exc: Exception):
    error_reporter.python_error()

    return JSONResponse(status_code=500, content={"message": "Internal Server Error"})


@api_router.post("/upload_image")
async def upload_image(image: UploadFile = File(...)):
    global request_id
    request_id = str(uuid.uuid4())

    # Read the image file
    image_bytes = await image.read()

    # Generate a unique name for the image based on the request_id and original filename
    destination_filename = f"{image.filename}"

    # Define the directory where to save the image
    image_dir = Path("src/scratch/dreambooth/data/users") / token

    # Create the directory if it does not exist
    image_dir.mkdir(parents=True, exist_ok=True)

    # Save the image locally
    with open(image_dir / destination_filename, "wb") as buffer:
        buffer.write(image_bytes)

    return {"image_url": str(image_dir / destination_filename)}


@api_router.post("/train")
async def train(input: UserAlbumInput):
    try:
        global model
        del model
    except:
        pass
    torch.cuda.empty_cache()

    # Log to BigQuery
    input_log = {
        "input_id": request_id,
        "user_id": input.user_id,
        "model": input.model,
        "song_name": input.song_name,
        "artist_name": input.artist_name,
        "album_name": input.album_name,
        "genre": input.genre,
        "lyric": input.lyric,
        "gender": input.gender,
        "image_urls": input.image_urls,
        "create_date": datetime.utcnow().astimezone(timezone("Asia/Seoul")).isoformat(),
    }
    bigquery_logger.log(input_log, "input")

    seeds = np.random.randint(100000)
    # Run the train.py script as a separate process
    process = subprocess.Popen(
        [
            "python",
            "src/scratch/dreambooth/run.py",
            "--config-file",
            "src/scratch/dreambooth/dreambooth.yaml",
            "--token",
            token,
            "--user-gender",
            input.gender,
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

    os.chdir("src/scratch")
    subprocess.run(command, shell=True)
    os.chdir("/opt/ml/level3_cv_finalproject-cv-03")
    return {"status": "Train started", "message": command}


@api_router.post("/inference")
async def inference(input: UserAlbumInput):
    summarization = get_dreambooth_prompt(
        input.lyric,
        input.album_name,
        input.song_name,
        input.gender,
        input.genre,
        input.artist_name,
    )

    prompt = f"A image of {summarization} music album cover with song title {input.song_name} by {input.artist_name}.\
        a {token} {input.gender} is in image."

    # Run the train.py script as a separate process
    process = subprocess.Popen(
        [
            "python",
            "src/scratch/dreambooth/inference.py",
            "--token",
            token,
            "--prompt",
            prompt,
            "--user-gender",
            input.gender,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Get the output and error messages from the process
    stdout, stderr = process.communicate()

    # Check if the process completed successfully
    if process.returncode != 0:
        return {"status": "error", "message": stderr.decode()}

    saved_dir = f"src/scratch/dreambooth/data/results/{token}"

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
    image_urls = gcs_uploader.save_image_to_gcs(urls)

    # Log to BigQuery
    output_id = str(uuid.uuid4())
    output_log = {
        "output_id": output_id,
        "input_id": request_id,
        "image_urls": image_urls,
        "seeds": [],  # TODO: 드림부스는 inference할때 시드값이 없나요?
        "prompt": prompt,
        "create_date": datetime.utcnow().astimezone(timezone("Asia/Seoul")).isoformat(),
    }
    bigquery_logger.log(output_log, "output")

    # return {"images": images}
    return {"images": image_urls, "output_id": output_id}


app.include_router(api_router)
