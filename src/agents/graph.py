import json
from collections.abc import AsyncIterator
from typing import Any, Protocol, cast

from langchain_core.embeddings import Embeddings
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send

from agents.common.agent import IAgent
from agents.common.constants import (
    COMMON,
    CONTINUE,
    FINALIZER,
    GATEKEEPER,
    INITIAL_SUMMARIZATION,
    MESSAGES,
    MESSAGES_SUMMARY,
    NEXT,
    SUBTASKS,
    SUMMARIZATION,
)
from agents.common.data import Message
from agents.common.response_converter import ResponseConverter
from agents.common.state import (
    CompanionState,
    GatekeeperResponse,
    Plan,
    SubTask,
    UserInput,
)
from agents.common.utils import filter_valid_messages, should_continue
from agents.k8s.agent import K8S_AGENT, KubernetesAgent
from agents.kyma.agent import KYMA_AGENT, KymaAgent
from agents.prompts import (
    COMMON_QUESTION_PROMPT,
    GATEKEEPER_INSTRUCTIONS,
    GATEKEEPER_PROMPT,
)
from agents.summarization.summarization import MessageSummarizer
from agents.supervisor.agent import SUPERVISOR, SupervisorAgent
from agents.supervisor.prompts import FINALIZER_PROMPT, FINALIZER_PROMPT_FOLLOW_UP
from services.k8s import IK8sClient
from utils.chain import ainvoke_chain
from utils.logging import get_logger
from utils.models.factory import IModel
from utils.settings import (
    MAIN_MODEL_MINI_NAME,
    MAIN_MODEL_NAME,
    SUMMARIZATION_TOKEN_LOWER_LIMIT,
    SUMMARIZATION_TOKEN_UPPER_LIMIT,
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


class IGraph(Protocol):
    """Graph interface."""

    def astream(
        self, conversation_id: str, message: Message, k8s_client: IK8sClient
    ) -> AsyncIterator[str]:
        """Stream the output to the caller asynchronously."""
        ...

    async def aget_messages(self, conversation_id: str) -> list[BaseMessage]:
        """Get messages from the graph state."""
        ...


class CompanionGraph:
    """Companion graph class. Represents all the workflow of the application."""

    models: dict[str, IModel | Embeddings]
    memory: BaseCheckpointSaver
    supervisor_agent: IAgent
    kyma_agent: IAgent
    k8s_agent: IAgent
    members: list[str] = []

    plan_parser = PydanticOutputParser(pydantic_object=Plan)

    planner_prompt: ChatPromptTemplate

    def __init__(
        self,
        models: dict[str, IModel | Embeddings],
        memory: BaseCheckpointSaver,
        handler: Any = None,
    ):
        self.models = models
        self.memory = memory
        self.handler = handler

        main_model_mini = models[MAIN_MODEL_MINI_NAME]
        main_model = models[MAIN_MODEL_NAME]

        self.kyma_agent = KymaAgent(models)

        self.k8s_agent = KubernetesAgent(cast(IModel, main_model))
        self.supervisor_agent = SupervisorAgent(
            models,
            members=[KYMA_AGENT, K8S_AGENT, COMMON],
        )

        self.summarization = MessageSummarizer(
            model=main_model_mini,
            tokenizer_model_name=MAIN_MODEL_NAME,
            token_lower_limit=SUMMARIZATION_TOKEN_LOWER_LIMIT,
            token_upper_limit=SUMMARIZATION_TOKEN_UPPER_LIMIT,
            messages_key=MESSAGES,
            messages_summary_key=MESSAGES_SUMMARY,
        )

        self.response_converter = ResponseConverter()

        self.members = [self.kyma_agent.name, self.k8s_agent.name, COMMON]
        self._common_chain = self._create_common_chain(cast(IModel, main_model_mini))
        self._gatekeeper_chain = self._create_gatekeeper_chain(
            cast(IModel, main_model_mini)
        )
        self.graph = self._build_graph()

    @staticmethod
    def _create_common_chain(model: IModel) -> RunnableSequence:
        """Common node chain to handle general queries."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", COMMON_QUESTION_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        return prompt | model.llm  # type: ignore

    async def _invoke_common_node(self, state: CompanionState) -> str:
        """Invoke the common node."""
        response = await ainvoke_chain(
            self._common_chain,
            {"messages": filter_valid_messages(state.get_messages_including_summary())},
        )
        return str(response.content)

    async def _common_node(self, state: CompanionState) -> dict[str, Any]:
        """Common node to handle general queries."""

        try:
            response = await self._invoke_common_node(state)
            return {
                MESSAGES: [
                    AIMessage(
                        content=response,
                        name=COMMON,
                    )
                ],
                SUBTASKS: state.subtasks,
            }
        except Exception:
            logger.exception("Error in common node")
            return {
                MESSAGES: [
                    AIMessage(
                        content="Sorry, I am unable to process the request.",
                        name=COMMON,
                    )
                ],
                SUBTASKS: state.subtasks,
            }

    @staticmethod
    def _create_gatekeeper_chain(model: IModel) -> RunnableSequence:
        """Gatekeeper node chain to handle general queries
        and queries that can answered from conversation history."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", GATEKEEPER_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
                ("system", GATEKEEPER_INSTRUCTIONS),
            ]
        )
        return prompt | model.llm.with_structured_output(GatekeeperResponse, method="function_calling")  # type: ignore

    async def _invoke_gatekeeper_node(
        self, state: CompanionState
    ) -> GatekeeperResponse | Any:
        """Invoke the Gatekeeper node."""
        response = await ainvoke_chain(
            self._gatekeeper_chain,
            {
                "messages": filter_valid_messages(
                    state.get_messages_including_summary()
                ),
            },
        )
        return response

    async def _gatekeeper_node(self, state: CompanionState) -> dict[str, Any]:
        """Gatekeeper node to handle general and queries that can answered from conversation history."""

        try:
            response = await self._invoke_gatekeeper_node(state)

            if response.forward_query:
                logger.debug("Gatekeeper node forwarding the query")
                return {
                    NEXT: SUPERVISOR,
                    SUBTASKS: [],
                }
            logger.debug("Gatekeeper node responding directly")
            return {
                NEXT: END,
                MESSAGES: [
                    AIMessage(
                        content=response.direct_response,
                        name=GATEKEEPER,
                    )
                ],
                SUBTASKS: [],
            }
        except Exception:
            logger.exception("Error in gatekeeper node")
            return {
                NEXT: END,
                MESSAGES: [
                    AIMessage(
                        content="Sorry, I am unable to process the request.",
                        name=GATEKEEPER,
                    )
                ],
                SUBTASKS: [],
            }

    def _get_members_str(self) -> str:
        return ", ".join(self.members)

    def _final_response_chain(self, state: CompanionState) -> RunnableSequence:
        # last human message must be the query
        last_human_message = next(
            (msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)),
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", FINALIZER_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
                ("system", FINALIZER_PROMPT_FOLLOW_UP),
            ]
        ).partial(members=self._get_members_str(), query=last_human_message.content)
        return prompt | self.model.llm  # type: ignore

    async def _generate_final_response(self, state: CompanionState) -> dict[str, Any]:
        """Generate the final response."""

        # If all required agents failed: tell user that we can't give them response due to agent failure
        if state.subtasks and all(subtask.is_error() for subtask in state.subtasks):
            return {
                MESSAGES: [
                    AIMessage(
                        content="We're unable to provide a response at this time due to agent failure. "
                        "Please try again or reach out to our support team for further assistance.",
                        name=FINALIZER,
                    )
                ],
                NEXT: END,
            }

        final_response_chain = self._final_response_chain(state)

        final_response = await ainvoke_chain(
            final_response_chain,
            {"messages": filter_valid_messages(state.messages)},
        )
        logger.debug("Final response generated")
        return {
            MESSAGES: [
                AIMessage(
                    content=final_response.content,
                    name=FINALIZER,
                )
            ],
            NEXT: END,
        }

    async def _get_converted_final_response(
        self, state: CompanionState
    ) -> dict[str, Any]:
        """Convert the generated final response"""
        try:
            final_response = await self._generate_final_response(state)
            logger.debug("Response conversion node started")
            return self.response_converter.convert_final_response(final_response)
        except Exception:
            logger.exception("Error in generating final response")
            return {
                MESSAGES: [
                    AIMessage(
                        content="Sorry, I encountered an error while processing the request. Try again later.",
                        name=FINALIZER,
                    )
                ]
            }

    async def _supervisor_node(self, state: CompanionState) -> Any:
        """Supervisor node to handle the conversation."""
        response = await self.supervisor_agent.agent_node().ainvoke(state)

        if response["next"] != FINALIZER:
            # only send the subtask message to the dedicated agent
            return Command(
                goto=Send(
                    response["next"],
                    {
                        "agent_messages": [
                            HumanMessage(content=response["messages"][-1].content)
                        ],
                        "k8s_client": state.k8s_client,
                    },
                ),
            )

        # finalizer needs all the messages to generate the final response
        return Command(update={"messages": state.messages}, goto=FINALIZER)

    def _build_graph(self) -> CompiledStateGraph:
        """Create the companion parent graph."""

        # Define a new graph.
        workflow = StateGraph(CompanionState)

        workflow.add_node(GATEKEEPER, self._gatekeeper_node)
        workflow.add_node(SUMMARIZATION, self.summarization.summarization_node)
        workflow.add_node(INITIAL_SUMMARIZATION, self.summarization.summarization_node)

        # Define the nodes of the graph.
        workflow.add_node(
            SUPERVISOR,
            self._supervisor_node,
            destinations=tuple(self.members + [FINALIZER]),
        )
        workflow.add_node(KYMA_AGENT, self.kyma_agent.agent_node())
        workflow.add_node(K8S_AGENT, self.k8s_agent.agent_node())
        workflow.add_node(COMMON, self._common_node)
        workflow.add_node(FINALIZER, self._generate_final_response)

        # Define the edges: (KymaAgent | KubernetesAgent | Common) --> summarization --> supervisor
        # The agents ALWAYS "report back" to the supervisor through summarization node.
        for member in self.members:
            workflow.add_edge(member, SUMMARIZATION)

        # Set the entrypoint: ENTRY --> Initial_Summarization
        workflow.set_entry_point(INITIAL_SUMMARIZATION)

        # Define the edges: Initial_Summarization --> Gatekeeper
        workflow.add_edge(INITIAL_SUMMARIZATION, GATEKEEPER)

        # Define the dynamic conditional edges: Gatekeeper --> (SUPERVISOR | END)
        workflow.add_conditional_edges(
            GATEKEEPER,
            lambda x: x.next,
            {
                SUPERVISOR: SUPERVISOR,
                END: END,
            },
        )

        # The supervisor dynamically populates the "next" field in the graph.
        # conditional_map: dict[Hashable, str] = {k: k for k in self.members + [END]}
        # # Define the dynamic conditional edges: supervisor --> (KymaAgent | KubernetesAgent | Common | END)
        # workflow.add_conditional_edges(SUPERVISOR, lambda x: x.next, conditional_map)

        workflow.add_conditional_edges(
            SUMMARIZATION,
            should_continue,
            {
                CONTINUE: SUPERVISOR,
                END: END,
            },
        )

        workflow.add_edge(FINALIZER, END)

        # Compile the graph.
        graph = workflow.compile(checkpointer=self.memory)

        return graph

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

        x_cluster_url = k8s_client.get_api_server()
        cluster_id = x_cluster_url.split(".")[1]

        async for chunk in self.graph.astream(
            input={
                "messages": messages,
                "input": user_input,
                "k8s_client": k8s_client,
                "subtasks": [],
                "error": None,
            },
            config={
                "configurable": {
                    "thread_id": conversation_id,
                },
                "callbacks": [
                    self.handler,
                ],
                "tags": [
                    cluster_id
                ],  # cluster_id as a tag for traceability and rate limiting
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
