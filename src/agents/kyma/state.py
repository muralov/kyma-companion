from collections.abc import Sequence
from typing import Annotated, Any, cast

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from utils.utils import to_sequence_messages


class KymaAgentState(BaseModel):
    """The state of the Kyma agent."""

    agent_messages: Annotated[Sequence[BaseMessage], add_messages]
    agent_messages_summary: str = ""
    k8s_client: Annotated[Any, Field(default=None, exclude=True)]

    # Subgraph private fields
    is_last_step: IsLastStep = Field(default=False)
    remaining_steps: RemainingSteps = Field(default=25)
    error: str | None = None

    # Model config for pydantic.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_messages_including_summary(self) -> Sequence[BaseMessage]:
        """Get messages including the summary message."""
        if self.agent_messages_summary:
            return to_sequence_messages(
                add_messages(
                    SystemMessage(content=self.agent_messages_summary),
                    cast(
                        list[
                            BaseMessage
                            | list[str]
                            | tuple[str, str]
                            | str
                            | dict[str, Any]
                        ],
                        self.agent_messages,
                    ),
                )
            )
        return self.agent_messages
