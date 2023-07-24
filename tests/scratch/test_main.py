# Built-in modules
import os
from tests.scratch.conftest import client


# Import your main FastAPI app object -- will refactor
from src.scratch.main import app

# Import TestClient -- will refactor
from fastapi.testclient import TestClient

# Pytorch
import torch
from torch import cuda

# Pytest
import pytest


device = "cuda" if cuda.is_available() else "cpu"


# Create a fixture for your client
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# src.scratch.main - test to see if generate_cover works well
def test_create_model_and_generate_image_and_gcs_upload(client):
    response = client.post(
        "/generate_cover",
        json={
            "song_names": "Test: song_names(generate_cover)",
            "artist_name": "Test: artist_name(generate_cover)",
            "genre": "Test: genre(generate_cover)",
            "album_name": "Test: album_name(generate_cover)",
            "lyric": "Test: lyric(generate_cover)",
        },
    )

    assert response.status_code == 200
    assert "images" in response.json().keys()  # Verify that "images" is in dictionary
    assert (
        len(response.json()["images"]) > 0
    )  # Verify that the image you created is stored well


# src.scratch.main - test to see if review works well
def test_get_review_and_gcs_upload(client):
    response = client.post(
        "/review", json={"rating": -1, "comment": "Test: comment(review)"}
    )

    assert response.status_code == 200


# src.scratch.main - test exception handling for good operation
def test_get_review_and_gcs_upload(client):
    pass
