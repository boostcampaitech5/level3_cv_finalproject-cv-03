# Python built-in modules
import io
import os
import base64
import uuid
from datetime import datetime

# Backend
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from pydantic import BaseModel

# Celery
from celery import Celery

# Other modules
import numpy as np
from pytz import timezone

# User Defined modules
from gcp.bigquery import BigQueryLogger
from gcp.error import ErrorReporter
from utils import load_yaml


# Load config
gcp_config = load_yaml(os.path.join("config", "private.yaml"), "gcp")
redis_config = load_yaml(os.path.join("config", "private.yaml"), "redis")
celery_config = load_yaml(os.path.join("config", "public.yaml"), "celery")
public_config = load_yaml(os.path.join("config", "public.yaml"))

# Start fastapi
app = FastAPI()

# Initialize Celery
celery_app = Celery(
    "tasks",
    broker=redis_config["redis_server_ip"],
    backend=redis_config["redis_server_ip"],
)

bigquery_logger = BigQueryLogger(gcp_config)
error_reporter = ErrorReporter(gcp_config)


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
async def generate_cover(album: AlbumInput):
    # Generate a unique ID for this request
    global request_id
    request_id = str(uuid.uuid4())

    # Request time
    request_time = datetime.utcnow().astimezone(timezone("Asia/Seoul")).isoformat()

    # Push task to the Celery queue
    task = celery_app.send_task("generate_cover", args=[album.dict(), request_id])

    # Get result (this will block until the task is done)
    task_result = task.get()

    album_log = {
        "request_id": request_id,
        "request_time": request_time,
        "song_names": album.song_names,
        "artist_name": album.artist_name,
        "genre": album.genre,
        "album_name": album.album_name,
        "lyric": album.lyric,
        "summarization": task_result["summarization"],
        "image_urls": task_result["image_urls"],
        "language": public_config["language"],
    }

    # Log to BigQuery
    bigquery_logger.log(album_log, "user_album")

    return {"images": task_result["image_urls"]}


# REST API - Post ~/review
@app.post("/review")
async def review(review: ReviewInput):
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
