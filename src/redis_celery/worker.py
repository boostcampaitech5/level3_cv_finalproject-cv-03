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

# Celery
from celery import Celery
from celery import signals

# User Defined modules
from .model import AlbumModel
from .gpt3_api import get_description
from .gcp.cloud_storage import GCSUploader
from .utils import load_yaml


# Load config
gcp_config = load_yaml(os.path.join("src/redis_celery/config", "private.yaml"), "gcp")
redis_config = load_yaml(
    os.path.join("src/redis_celery/config", "private.yaml"), "redis"
)
celery_config = load_yaml(
    os.path.join("src/redis_celery/config", "public.yaml"), "celery"
)
public_config = load_yaml(os.path.join("src/redis_celery/config", "public.yaml"))


# f'redis://{redis_config["host"]}:{redis_config["port"]}/{redis_config["db"]}'
# Initialize Celery
celery_app = Celery(
    "tasks",
    broker=redis_config["redis_server_ip"],
    backend=redis_config["redis_server_ip"],
)
celery_app.conf.broker_connection_retry_on_startup = celery_config[
    "broker_connection_retry_on_startup"
]
celery_app.conf.worker_pool = celery_config["worker_pool"]

gcs_uploader = GCSUploader(gcp_config)


@signals.worker_process_init.connect
def setup_worker_init(*args, **kwargs):
    device = "cuda" if cuda.is_available() else "cpu"
    global model
    model = AlbumModel(public_config["model"], public_config["language"], device)
    model.get_model()


@celery_app.task(name="generate_cover")
def generate_cover(album, request_id):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = "cuda" if cuda.is_available() else "cpu"

    image_urls = []

    summarization = get_description(
        album["lyric"], album["artist_name"], album["album_name"], album["song_names"]
    )

    seeds = np.random.randint(
        public_config["generate"]["max_seed"], size=public_config["generate"]["n_gen"]
    )

    for i, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(int(seed))

        # Generate Images
        with torch.no_grad():
            image = model.pipeline(
                prompt=f"A photo of a {album['genre']} album cover with a {summarization} atmosphere visualized.",
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
        base64_str = base64.b64encode(byte_arr).decode()

        # Upload to GCS
        image_url = gcs_uploader.save_image_to_gcs(
            byte_arr,
            f"{request_id}_image_{i}.{public_config['generate']['save_format']}",
        )
        image_urls.append(image_url)

    return {
        "image_urls": image_urls,
        "summarization": summarization,
    }


# Start the worker
if __name__ == "__main__":
    celery_app.start()
