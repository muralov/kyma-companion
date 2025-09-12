from typing import Any, Literal, Protocol

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from agents.common.chunk_summarizer import (
    IToolResponseSummarizer,
    ToolResponseSummarizer,
)
from agents.common.constants import (
    CONTINUE,
    ERROR,
    MESSAGES_SUMMARY,
    SUMMARIZATION,
)
from agents.common.error_handler import (
    token_counting_error_handler,
    tool_parsing_error_handler,
)
from agents.common.exceptions import TotalChunksLimitExceededError
from agents.common.prompts import JOULE_CONTEXT_INFORMATION
from agents.common.state import BaseAgentState, SubTaskStatus
from agents.common.utils import (
    compute_string_token_count,
    convert_string_to_object,
    filter_valid_messages,
    should_continue,
)
from agents.kyma.state import KymaAgentState
from agents.summarization.summarization import MessageSummarizer
from utils.chain import ainvoke_chain
from utils.logging import get_logger
from utils.models.factory import IModel
from utils.settings import (
    GRAPH_STEP_TIMEOUT_SECONDS,
    SUMMARIZATION_TOKEN_LOWER_LIMIT,
    SUMMARIZATION_TOKEN_UPPER_LIMIT,
    TOOL_RESPONSE_TOKEN_COUNT_LIMIT,
    TOTAL_CHUNKS_LIMIT,
)

logger = get_logger(__name__)


AGENT_STEPS_NUMBER = 3


def agent_edge(state: KymaAgentState) -> Literal["Summarization", "finalizer"]:
    """Function that determines whether to call tools or finalizer."""
    last_message = state.messages[-1]
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        return "finalizer"
    return "Summarization"  # from SUMMARIZATION --> tools


class IAgent(Protocol):
    """Agent interface."""

    def agent_node(self) -> CompiledStateGraph:
        """Main agent function."""
        ...

    @property
    def name(self) -> str:
        """Agent name."""
        ...


class BaseAgent:
    """Abstract base agent class."""

    def __init__(
        self,
        name: str,
        model: IModel | Embeddings,
        tools: list,
        agent_prompt: ChatPromptTemplate,
        state_class: type,
    ):
        self._name = name
        self.model = model
        self.tools = tools
        self.summarization = MessageSummarizer(
            model=model,
            tokenizer_model_name=model.name,
            token_lower_limit=SUMMARIZATION_TOKEN_LOWER_LIMIT,
            token_upper_limit=SUMMARIZATION_TOKEN_UPPER_LIMIT,
            messages_key="messages",
            messages_summary_key=MESSAGES_SUMMARY,
        )
        self.tool_response_summarization: IToolResponseSummarizer = (
            ToolResponseSummarizer(model=model)
        )
        self.chain = self._create_chain(agent_prompt)
        self.graph = self._build_graph(state_class)
        self.graph.step_timeout = GRAPH_STEP_TIMEOUT_SECONDS

    @property
    def name(self) -> str:
        """Agent name."""
        return self._name

    def agent_node(self) -> CompiledStateGraph:
        """Get agent node function."""
        return self.graph

    def _create_chain(self, agent_prompt: ChatPromptTemplate) -> Any:
        return agent_prompt | self.model.llm.bind_tools(self.tools)

    async def _invoke_chain(
        self,
        state: KymaAgentState,
        config: RunnableConfig,
        tool_summarized_response: str | None = "",
    ) -> Any:
        agent_messages = state.get_messages_including_summary()

        # Append the tool summarized tool response
        filtered_agent_messages = filter_valid_messages(agent_messages)
        if tool_summarized_response:
            filtered_agent_messages.append(
                AIMessage(
                    content="Summarized Tool Response - " + tool_summarized_response
                )
            )
        # invoke the chain.
        response = await ainvoke_chain(
            self.chain,
            {
                "messages": filtered_agent_messages,
            },
            config=config,
        )
        return response

    @tool_parsing_error_handler
    def _parse_tool_message(self, message_content: str) -> Any:
        """Parse tool message content into an object."""
        return convert_string_to_object(message_content)

    @token_counting_error_handler
    def _compute_token_count(self, content: str) -> int:
        """Compute token count for the given content."""
        return compute_string_token_count(content, self.model.name)

    async def _execute_summarization(
        self,
        tool_responses: list[Any],
        user_query: str,
        config: RunnableConfig,
        num_chunks: int,
    ) -> str:
        """Execute the actual summarization process."""
        return await self.tool_response_summarization.summarize_tool_response(
            tool_response=tool_responses,
            user_query=user_query,
            config=config,
            nums_of_chunks=num_chunks,
        )

    async def _summarize_tool_response(
        self, state: KymaAgentState, config: RunnableConfig
    ) -> str:
        """
        Summarize tool responses if they exceed the token limit.

        This method processes tool messages from the agent state, checks if their
        combined token count exceeds the model's limit, and if so, summarizes them
        using chunked summarization to reduce token usage.
        """

        # Extract tool responses from recent messages (in reverse order)
        tool_responses: list[Any] = []

        for message in reversed(state.messages):
            if isinstance(message, ToolMessage):
                # Use decorated method for parsing with error handling
                tool_response_object = self._parse_tool_message(str(message.content))
                if tool_response_object is not None:  # None indicates parsing failed
                    if isinstance(tool_response_object, list):
                        tool_responses.extend(tool_response_object)
                    else:
                        tool_responses.append(tool_response_object)
            else:
                # Stop when we hit a non-tool message
                break

        # Early return if no tool responses found
        if not tool_responses:
            return ""

        token_count = self._compute_token_count(str(tool_responses))
        logger.info(f"Tool Response Token count: {token_count}")

        model_token_limit = TOOL_RESPONSE_TOKEN_COUNT_LIMIT
        if token_count <= model_token_limit:
            logger.debug("Tool response within token limit, no summarization needed")
            return ""

        # Calculate number of chunks needed
        num_chunks = (token_count // model_token_limit) + 1
        logger.info(f"Number of chunks for summarization: {num_chunks}")

        # Validate chunk limit
        if num_chunks > TOTAL_CHUNKS_LIMIT:
            logger.error(
                f"Tool response requires {num_chunks} chunks, "
                f"which exceeds the limit of {TOTAL_CHUNKS_LIMIT}"
            )
            raise TotalChunksLimitExceededError()

        # Perform summarization using decorated method
        summarized_response = await self._execute_summarization(
            tool_responses=tool_responses,
            user_query=str(state.messages[-1].content),
            config=config,
            num_chunks=num_chunks,
        )

        # Update processed tool messages to indicate they've been summarized
        self._mark_tool_messages_as_summarized(state)

        return str(summarized_response)

    def _mark_tool_messages_as_summarized(self, state: KymaAgentState) -> None:
        """
        Mark the specified number of recent tool messages as summarized.

        Args:
            state: The agent state containing messages to update
        """
        for message in reversed(state.messages):
            if isinstance(message, ToolMessage):
                message.content = "Summarized"
            elif not isinstance(message, ToolMessage):
                # Stop when we hit a non-tool message
                break

    async def _summarize_tool_response_with_error_handling(
        self, state: KymaAgentState, config: RunnableConfig
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Summarize tool response with error handling. This method encapsulates the
        error handling logic for the summarization process.
        """
        try:
            summarized_tool_response = await self._summarize_tool_response(
                state, config
            )
            return summarized_tool_response, None
        except TotalChunksLimitExceededError:
            logger.exception("Error while summarizing the tool response.")
            if state.my_task:
                state.my_task.status = SubTaskStatus.COMPLETED
            error_dict: dict[str, Any] = {
                "messages": [
                    AIMessage(
                        content="Your request is too broad and requires analyzing "
                        "more resources than allowed at once. "
                        "Please specify a particular resource you'd like to analyze so "
                        f"I can assist you more effectively. {JOULE_CONTEXT_INFORMATION}",
                        name=self.name,
                    )
                ]
            }
            return "", error_dict
        except Exception:
            logger.exception("Error while summarizing the tool response.")
            err_response: dict[str, Any] = {
                "messages": [
                    AIMessage(
                        content="Sorry, an unexpected error occurred while processing your request. "
                        "Please try again later.",
                        name=self.name,
                    )
                ],
                ERROR: "An error occurred while processing the request",
            }
            return "", err_response

    def _handle_recursive_limit_error(self, state: KymaAgentState) -> dict[str, Any]:
        """Handle recursive limit error."""
        logger.error(
            f"Agent reached the recursive limit, steps remaining: {state.remaining_steps}."
        )
        return {
            "messages": [
                AIMessage(
                    content="Agent reached the recursive limit, not able to call Tools again",
                    name=self.name,
                )
            ],
        }

    async def _invoke_chain_with_error_handling(
        self,
        state: KymaAgentState,
        config: RunnableConfig,
        summarized_tool_response: str = "",
    ) -> tuple[Any, dict[str, Any] | None]:
        """Handle model node error."""
        try:
            response = await self._invoke_chain(state, config, summarized_tool_response)
            return response, None
        except Exception:
            logger.exception("An error occurred while processing the request.")
            error_response = {
                "messages": [
                    AIMessage(
                        content="Sorry, an unexpected error occurred while processing your request. "
                        "Please try again later.",
                        name=self.name,
                    )
                ],
                ERROR: "An error occurred while processing the request",
            }
            return None, error_response

    async def _model_node(
        self, state: KymaAgentState, config: RunnableConfig
    ) -> dict[str, Any]:
        # if the recursive limit is reached, return a message.
        if state.remaining_steps <= AGENT_STEPS_NUMBER:
            return self._handle_recursive_limit_error(state)

        # if the last message is a tool message, summarize the tool response if needed.
        summarized_tool_response = ""
        if state.messages and isinstance(state.messages[-1], ToolMessage):
            summarized_tool_response, error_response = (
                await self._summarize_tool_response_with_error_handling(state, config)
            )
            if error_response:
                return error_response

        response, error_response = await self._invoke_chain_with_error_handling(
            state, config, summarized_tool_response
        )
        if error_response:
            return error_response

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
                        content="Sorry, I need more steps to process the request.",
                        name=self.name,
                    )
                ],
            }

        response.additional_kwargs["owner"] = self.name
        return {
            "messages": [response],
        }

    def _finalizer_node(self, state: KymaAgentState, config: RunnableConfig) -> Any:
        """Finalizer node will mark the task as completed."""

        state.messages[-1].content = (
            f"'{state.messages[0].content}' , Agent Response - {state.messages[-1].content or ''}"
        )

        # clean all agent messages to avoid populating the checkpoint with unnecessary messages.
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=state.messages[-1].content,
                        name=self.name,
                        id=state.messages[-1].id,
                    )
                ],
            },
            goto=SUMMARIZATION,
            # goto=Send(
            #     SUMMARIZATION,
            #     {
            #         "messages": [
            #             AIMessage(
            #                 content=state.messages[-1].content,
            #                 name=self.name,
            #                 id=state.messages[-1].id,
            #             )
            #         ],
            #     },
            # ),
            graph=Command.PARENT,
        )

    def _build_graph(self, state_class: type) -> CompiledStateGraph:
        # Define a new graph
        workflow: StateGraph = StateGraph(state_class)

        # Define nodes with async awareness
        workflow.add_node("agent", self._model_node)
        workflow.add_node("tools", ToolNode(tools=self.tools))
        workflow.add_node("finalizer", self._finalizer_node)
        workflow.add_node(SUMMARIZATION, self.summarization.summarization_node)

        # Set the entrypoint: ENTRY --> agent
        workflow.set_entry_point("agent")

        # Define the edge: agent --> (summarization | finalizer)
        workflow.add_conditional_edges("agent", agent_edge)

        # Define the edge: tool --> tool
        workflow.add_edge("tools", "agent")

        # Define the edge: summarization --> agent | error_handler
        workflow.add_conditional_edges(
            SUMMARIZATION,
            should_continue,
            {
                CONTINUE: "tools",
                END: END,
            },
        )

        # Define the edge: finalizer --> END
        workflow.add_edge("finalizer", "__end__")

        return workflow.compile()
