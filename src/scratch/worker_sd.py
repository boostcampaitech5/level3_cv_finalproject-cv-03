# Python Built-in modules
import os
import io
import base64

# Pytorch
import torch
from torch import cuda

# ETC
from PIL import Image
import numpy as np
from numpy import random
from pytz import timezone
from datetime import datetime

# Celery
from celery import Celery
from celery import signals

# User Defined modules
from .model import StableDiffusion
from .gpt3_api import get_description
from .gcp.cloud_storage import GCSUploader
from .gcp.bigquery import BigQueryLogger
from .utils import load_yaml


# Load config
gcp_config = load_yaml(os.path.join("src/scratch/config", "private.yaml"), "gcp")
redis_config = load_yaml(os.path.join("src/scratch/config", "private.yaml"), "redis")
celery_config = load_yaml(os.path.join("src/scratch/config", "public.yaml"), "celery")
public_config = load_yaml(os.path.join("src/scratch/config", "public.yaml"))


# Initialize Celery
celery_app = Celery(
    "tasks",
    broker=redis_config["redis_server_ip"],
    backend=redis_config["redis_server_ip"],
)

celery_app.conf.worker_pool = "solo"

gcs_uploader = GCSUploader(gcp_config)
bigquery_logger = BigQueryLogger(gcp_config)


@signals.worker_process_init.connect
def setup_worker_init(*args, **kwargs):
    device = "cuda" if cuda.is_available() else "cpu"
    global model
    model = AlbumModel(public_config["model"], public_config["language"], device)
    model.get_model()


@celery_app.task(name="generate_cover")
def generate_cover(input, request_id):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = "cuda" if cuda.is_available() else "cpu"

    # Request time
    request_time = datetime.utcnow().astimezone(timezone("Asia/Seoul")).isoformat()

    images = []
    urls = []

    summarization = get_description(
        input["lyric"], input["artist_name"], input["album_name"], input["song_name"]
    )

    seeds = np.random.randint(
        public_config["generate"]["max_seed"], size=public_config["generate"]["n_gen"]
    )

    # Inference Prompt
    prompt = f"A photo of a {input['genre']} album cover with a {summarization} atmosphere visualized."

    for i, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(int(seed))

        # Generate Images
        with torch.no_grad():
            image = model.pipeline(
                prompt=prompt,
                num_inference_steps=20,
                generator=generator,
            ).images[0]

        image = image.resize(
            (public_config["generate"]["height"], public_config["generate"]["width"])
        )

        # Convert to base64-encoded string
        byte_arr = io.BytesIO()
        image.save(byte_arr, format=public_config["generate"]["save_format"])
        byte_arr = byte_arr.getvalue()

        # Upload to GCS
        urls.append(
            [
                byte_arr,
                f"{request_id}_image_{i}.{public_config['generate']['save_format']}",
            ]
        )
        images.append(byte_arr)

    # Upload to GCS
    image_urls = gcs_uploader.save_image_to_gcs(urls)

    input_log = {
        "input_id": request_id,
        "user_id": input["user_id"],
        "model": input["model"],
        "song_name": input["song_name"],
        "artist_name": input["artist_name"],
        "album_name": input["album_name"],
        "genre": input["genre"],
        "lyric": input["lyric"],
        "gender": input["gender"],
        "image_urls": input["image_urls"],
        "create_date": datetime.utcnow().astimezone(timezone("Asia/Seoul")).isoformat(),
    }
    bigquery_logger.log(input_log, "input")

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


# Start the worker
if __name__ == "__main__":
    celery_app.start()
