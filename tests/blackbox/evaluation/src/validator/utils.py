from tests.blackbox.evaluation.src.common.config import Config
from tests.blackbox.evaluation.src.validator.enums import AIModel
from tests.blackbox.evaluation.src.validator.validator import ValidatorInterface, ChatOpenAIValidator


def create_validator(config: Config) -> ValidatorInterface:
    if config.model_name == AIModel.CHATGPT_4_O:
        return ChatOpenAIValidator(config)

    raise ValueError(f"Unsupported model name: {config.model_name}")
