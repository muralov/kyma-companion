from shared_kernel.logging import get_logger
from shared_kernel.models import get_model

logger = get_logger(__name__)


async def init_chat() -> dict:
    """ Initialize the chat """
    logger.info("Initializing chat...")
    return {"message": "Chat is initialized!"}


async def handle_request() -> dict:
    """ Chat with the Kyma companion """
    logger.info("Processing request...")

    llm = get_model("gpt-4o")
    logger.info(f"LLM model {llm} is created.")

    return {"message": "Hello I am Kyma Companion!"}
