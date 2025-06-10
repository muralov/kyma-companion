import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from langchain_core.embeddings import Embeddings
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command, Send

from agents.common.data import Message
from agents.common.new_agent import NewAgent
from agents.common.state import (
    NewCompanionState,
    SubTask,
    UserInput,
)
from agents.k8s.tools.logs import fetch_pod_logs_tool
from agents.k8s.tools.query import k8s_query_tool
from agents.kyma.tools.query import fetch_kyma_resource_version, kyma_query_tool
from agents.kyma.tools.search import SearchKymaDocTool
from services.k8s import IK8sClient
from utils.logging import get_logger
from utils.models.factory import IModel
from utils.settings import (
    MAIN_MODEL_NAME,
)

logger = get_logger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for AIMessage, HumanMessage, and SubTask.
    Default JSON cannot serialize these objects.
    """

    def default(self, o):  # noqa D102
        """Custom JSON encoder for RemoveMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage, and SubTask."""
        if isinstance(
            o,
            RemoveMessage
            | AIMessage
            | HumanMessage
            | SystemMessage
            | ToolMessage
            | SubTask,
        ):
            return o.__dict__
        elif isinstance(o, IK8sClient):
            return o.model_dump()
        return super().default(o)


def create_task_description_handoff_tool(
    *, agent_name: str, description: str | None = None
) -> BaseTool:
    """Create a tool to handoff a task to a given agent."""

    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        # this is populated by the supervisor LLM
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        # these parameters are ignored by the LLM
        state: Annotated[NewCompanionState, InjectedState],
        k8s_client: Annotated[IK8sClient, InjectedState],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {
            "messages": [task_description_message],
            "k8s_client": k8s_client,
        }
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool


assign_to_kyma_agent_with_description = create_task_description_handoff_tool(
    agent_name="kyma_agent",
    description="Assign task to a kyma agent.",
)

assign_to_k8s_agent_with_description = create_task_description_handoff_tool(
    agent_name="k8s_agent",
    description="Assign task to a k8s agent.",
)


class NewGraph:
    """New graph class. Represents all the workflow of the application."""

    def __init__(
        self,
        models: dict[str, IModel | Embeddings],
        memory: BaseCheckpointSaver,
        handler: Any = None,
    ):
        self.models = models
        self.memory = memory
        self.handler = handler

        self.kyma_agent = NewAgent(
            name="kyma_agent",
            model=models[MAIN_MODEL_NAME],
            system_prompt=(
                "You are a Kyma agent.\n\n"
                "INSTRUCTIONS:\n"
                "- Assist ONLY with Kyma-related tasks, DO NOT do any k8s-related tasks\n"
                "- After you're done with your tasks, respond to the supervisor directly\n"
                "- Respond ONLY with the results of your work, do NOT include ANY other text."
            ),
            tools=[
                fetch_kyma_resource_version,
                kyma_query_tool,
                SearchKymaDocTool(models),
            ],
        )

        self.k8s_agent = NewAgent(
            name="k8s_agent",
            model=models[MAIN_MODEL_NAME],
            system_prompt=(
                "You are a Kubernetes agent.\n\n"
                "INSTRUCTIONS:\n"
                "- Assist ONLY with Kubernetes-related tasks, DO NOT do any Kyma-related tasks\n"
                "- After you're done with your tasks, respond to the supervisor directly\n"
                "- Respond ONLY with the results of your work, do NOT include ANY other text."
            ),
            tools=[
                k8s_query_tool,
                fetch_pod_logs_tool,
            ],
        )

        self.supervisor_agent = create_react_agent(
            model=models[MAIN_MODEL_NAME].llm,
            tools=[
                assign_to_kyma_agent_with_description,
                assign_to_k8s_agent_with_description,
            ],
            prompt=(
                "You are a supervisor managing two agents:\n"
                "- a kyma agent. Assign kyma-related tasks to this assistant\n"
                "- a kubernetes agent. Assign kubernetes-related tasks to this assistant\n"
                "Assign work to one agent at a time, do not call agents in parallel.\n"
                "Do not do any work yourself."
            ),
            name="supervisor",
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        return (
            StateGraph(NewCompanionState)
            .add_node(
                self.supervisor_agent,
                destinations=("kyma_agent", "k8s_agent"),
            )
            .add_node("kyma_agent", self.kyma_agent.graph)
            .add_node("k8s_agent", self.k8s_agent.graph)
            .add_edge(START, "supervisor")
            .add_edge("kyma_agent", "supervisor")
            .add_edge("k8s_agent", "supervisor")
            .compile(checkpointer=self.memory)
        )

    async def astream(
        self, conversation_id: str, message: Message, k8s_client: IK8sClient
    ) -> AsyncIterator[str]:
        """Stream the output to the caller asynchronously."""
        user_input = UserInput(**message.__dict__)
        messages: list[BaseMessage] = [HumanMessage(content=message.query)]
        resource_context = user_input.get_resource_information()
        if resource_context and len(resource_context) > 0:
            messages.insert(
                0,
                SystemMessage(
                    content=f"The user query is related to: {resource_context}"
                ),
            )

        async for chunk in self.graph.astream(
            input={
                "messages": messages,
                "k8s_client": k8s_client,
            },
            config={
                "configurable": {
                    "thread_id": conversation_id,
                },
                "callbacks": [
                    self.handler,
                ],
            },
        ):
            chunk_json = json.dumps(chunk, cls=CustomJSONEncoder)
            if "__end__" not in chunk:
                yield chunk_json

    async def aget_messages(self, conversation_id: str) -> list[BaseMessage]:
        """Get messages from the graph state."""
        latest_state = await self.graph.aget_state(
            {
                "configurable": {
                    "thread_id": conversation_id,
                },
            }
        )
        if latest_state.values and "messages" in latest_state.values:
            return latest_state.values["messages"]  # type: ignore
        return []

    async def aget_thread_owner(self, conversation_id: str) -> str | None:
        """Get the owner of the thread."""
        state = await self.graph.aget_state(
            {
                "configurable": {
                    "thread_id": conversation_id,
                },
            }
        )
        if (
            state
            and state.values
            and "thread_owner" in state.values
            and state.values["thread_owner"] != ""
        ):
            return str(state.values["thread_owner"])
        return None

    async def aupdate_thread_owner(
        self, conversation_id: str, user_identifier: str
    ) -> None:
        """Update the owner of the thread."""
        await self.graph.aupdate_state(
            {
                "configurable": {
                    "thread_id": conversation_id,
                },
            },
            {
                "thread_owner": user_identifier,
            },
        )
