# Python built-in modules
import os
import io
import base64
import uuid
from datetime import datetime

# Backend
from fastapi import FastAPI, Request, Depends, APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Pydantic
from pydantic import BaseModel

# Celery
from celery import Celery
from celery.result import AsyncResult
import asyncio

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
from .gcp.bigquery import BigQueryLogger
from .gcp.error import ErrorReporter
from .utils import load_yaml

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load config
gcp_config = load_yaml(os.path.join("src/scratch/config", "private.yaml"), "gcp")
public_config = load_yaml(os.path.join("src/scratch/config", "public.yaml"))
train_config = load_yaml(os.path.join("src/scratch/dreambooth", "dreambooth.yaml"))
bigquery_config = gcp_config["bigquery"]


bigquery_logger = BigQueryLogger(gcp_config)
error_reporter = ErrorReporter(gcp_config)

# Start fastapi
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Allowed Methods
ALLOWED_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]

# Origins for CORS
origins = ["http://aibum.net", "http://34.22.72.143"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Celery
celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)


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
    rating: float
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
    # Generate a unique ID for this request(This will be shared with "/review")
    global request_id
    request_id = str(uuid.uuid4())

    # Push task to the Celery queue
    task = celery_app.send_task("generate_cover", args=[input.dict(), request_id])

    return {"task_id": task.id}


@api_router.get("/get_task_result/{task_id}")
async def get_task_result(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)

    if not task_result.ready():
        return {"status": str(task_result.status)}

    result = task_result.get()
    return {"status": str(task_result.status), "result": result}


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


@api_router.post("/upload_image")
async def upload_image(image: UploadFile = File(...)):
    global token, request_id
    request_id = str(uuid.uuid4())

    # Read the image file
    image_bytes = await image.read()
    image_content = base64.b64encode(image_bytes).decode()

    # Use asyncio.gather to run the task asynchronously
    task = celery_app.send_task(
        "save_image", args=[image.filename, image_content, token]
    )
    asyncio.create_task(
        wait_for_task_completion(task)
    )  # Run the task in the background
    return {"status": "File upload started"}


@api_router.post("/train_inference")
async def train(input: UserAlbumInput):
    # Use asyncio.gather to run the task asynchronously
    task = celery_app.send_task(
        "train_inference", args=[input.dict(), token, request_id]
    )

    return {"task_id": task.id}


# Helper function to wait for task completion
async def wait_for_task_completion(task):
    try:
        result = task.get()
        # Process the result if needed
        print(result)
    except asyncio.TimeoutError:
        # Task took too long to complete
        print("Task timed out.")
    except Exception as e:
        # Handle any other exceptions
        print("Error occurred:", e)


app.include_router(api_router)


# Exception handling using google cloud
@app.exception_handler(Exception)
async def handle_exceptions(request: Request, exc: Exception):
    error_reporter.python_error()

    return JSONResponse(status_code=500, content={"message": "Internal Server Error"})
