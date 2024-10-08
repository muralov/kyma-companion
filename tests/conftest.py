import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def set_test_environment():
    os.environ["REDIS_HOST"] = "127.0.0.1"
    os.environ["REDIS_URL"] = "http://127.0.0.1:6379/0"
    os.environ["REDIS_PORT"] = "6379"
    os.environ["LANGFUSE_SECRET_KEY"] = "test_secret_key"
    os.environ["LANGFUSE_PUBLIC_KEY"] = "test_public_key"
    os.environ["LANGFUSE_HOST"] = "test_langfuse_host"
