# Other modules
from fastapi.testclient import TestClient
import pytest

# Built-in modules
from src.scratch.main import app


@pytest.fixture(scope="session")
def client():
    return TestClient(app)
