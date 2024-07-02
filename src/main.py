import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes.api import app  # noqa: F401 - uvicorn needs to import the app

from domain.chat.services.chat import handle_request, init_chat
from shared_kernel.logging import get_logger

logger = get_logger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/chat/init")
async def init() -> dict:
    """ Endpoint to initialize the chat with the Kyma companion """
    return await init_chat()


@app.get("/chat")
async def chat() -> dict:
    """ Endpoint to chat with the Kyma companion """
    return await handle_request()


if __name__ == "__main__":
    config = uvicorn.Config(
        app="main:app",
        port=8000,
        log_level="info",
        use_colors=True,
    )
    server = uvicorn.Server(config)
    server.run()
