import operator
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.managed import IsLastStep
from langgraph.prebuilt import ToolNode

from agents.common.constants import MESSAGES
from agents.common.state import AgentState, SubTask, SubTaskStatus
from agents.common.utils import filter_messages
from agents.k8s.constants import (
    GRAPH_STEP_TIMEOUT_SECONDS,
    IS_LAST_STEP,
    K8S_AGENT,
    K8S_CLIENT,
    MY_TASK,
)
from agents.k8s.prompts import K8S_AGENT_PROMPT
from agents.k8s.tools.query import k8s_query_tool
from services.k8s import IK8sClient
from utils.logging import get_logger
from utils.models import IModel

logger = get_logger(__name__)


class KubernetesAgentState(BaseModel):
    """The state of the Kubernetes agent."""

    # Fields shared with the parent graph (Kyma graph).
    messages: Annotated[Sequence[BaseMessage], operator.add]
    subtasks: list[SubTask] | None = []
    k8s_client: IK8sClient

    # Subgraph private fields
    my_task: SubTask | None = None
    is_last_step: IsLastStep

    class Config:
        arbitrary_types_allowed = True
        fields = {K8S_CLIENT: {"exclude": True}}


class KubernetesAgent:
    """Kubernetes agent class."""

    _name: str = K8S_AGENT

    def __init__(self, model: IModel):
        self.tools = [k8s_query_tool]
        self.model = model.llm.bind_tools(self.tools)
        self.system_prompt = SystemMessage(K8S_AGENT_PROMPT)
        self.graph = self._build_graph()
        self.graph.step_timeout = GRAPH_STEP_TIMEOUT_SECONDS

    @property
    def name(self) -> str:
        """Agent name."""
        return self._name

    def agent_node(self) -> CompiledGraph:
        """Get Kubernetes agent node function."""
        return self.graph

    def _subtask_selector_node(self, state: KubernetesAgentState) -> dict[str, Any]:
        if state.k8s_client is None:
            raise ValueError("Kubernetes client is not initialized.")

        # find subtasks assigned to this agent and not completed.
        for subtask in state.subtasks:
            if (
                subtask.assigned_to == self.name
                and subtask.status != SubTaskStatus.COMPLETED
            ):
                return {
                    MY_TASK: subtask,
                }

        # if no subtask is found, return is_last_step as True.
        return {
            IS_LAST_STEP: True,
            MESSAGES: [
                AIMessage(
                    content="All my subtasks are already completed.",
                    name=self.name,
                )
            ],
        }

    def _model_node(
        self, state: KubernetesAgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        messages = (
            [self.system_prompt]
            + filter_messages(state.messages)
            + [HumanMessage(content=state.my_task.description)]
        )

        # invoke model.
        response = self.model.invoke(messages, config)
        # if the recursive limit is reached and the response is a tool call, return a message.
        # 'is_last_step' is a boolean that is True if the recursive limit is reached.
        if (
            state.is_last_step
            and isinstance(response, AIMessage)
            and response.tool_calls
        ):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, the kubernetes agent needs more steps to process the request.",
                    )
                ]
            }
        return {MESSAGES: [response]}

    def _build_graph(self) -> CompiledGraph:
        # Define a new graph
        workflow = StateGraph(KubernetesAgentState)

        # Define the nodes we will cycle between
        workflow.add_node("subtask_selector", self._subtask_selector_node)
        workflow.add_node("agent", self._model_node)
        workflow.add_node("tools", ToolNode(self.tools))

        # Set the entrypoint: ENTRY --> subtask_selector
        workflow.set_entry_point("subtask_selector")

        #
        # Define the edge: subtask_selector --> agent
        def is_any_subtask(state: KubernetesAgentState) -> Literal["agent", "__end__"]:
            if state.is_last_step and state.my_task is None:
                return "__end__"
            return "agent"

        # Add the conditional edge.
        workflow.add_conditional_edges("subtask_selector", is_any_subtask)

        # Define the edge: agent --> tool | end
        def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
            """Function that determines whether to continue or not."""
            # If there is no function call, then we finish
            last_message = state.messages[-1]
            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                return "__end__"
            return "tools"

        # Add the conditional edge.
        workflow.add_conditional_edges("agent", should_continue)

        # Define the edge: tool --> agent
        workflow.add_edge("tools", "agent")

        return workflow.compile()
