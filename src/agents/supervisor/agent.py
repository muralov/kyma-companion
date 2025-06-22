import json
from typing import Any, Literal, cast

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.common.constants import (
    COMMON,
    FINALIZER,
    K8S_AGENT,
    KYMA_AGENT,
    PLANNER,
)
from agents.common.exceptions import SubtasksMissingError
from agents.common.response_converter import IResponseConverter
from agents.common.state import Plan, Route
from agents.common.utils import (
    create_node_output,
    filter_messages,
    filter_valid_messages,
)
from agents.supervisor.prompts import (
    PLANNER_STEP_INSTRUCTIONS,
    PLANNER_SYSTEM_PROMPT,
    ROUTER_STEP_INSTRUCTIONS,
    ROUTER_SYSTEM_PROMPT,
)
from agents.supervisor.state import SupervisorState
from utils.chain import ainvoke_chain
from utils.filter_messages import (
    filter_messages_via_checks,
    is_ai_message,
    is_finalizer_message,
    is_human_message,
    is_system_message,
)
from utils.logging import get_logger
from utils.models.factory import IModel
from utils.settings import MAIN_MODEL_MINI_NAME

SUPERVISOR = "Supervisor"
ROUTER = "Router"

logger = get_logger(__name__)


def decide_route_or_exit(state: SupervisorState) -> Literal[ROUTER, END]:  # type: ignore
    """Return the next node whether to route or exit with a direct response."""
    if state.next == END:
        logger.debug("Ending the workflow.")
        return END
    # if there is a recoverable error
    if state.error:
        logger.error(f"Exiting the workflow due to the error: {state.error}")
        return END

    return ROUTER


def decide_entry_point(state: SupervisorState) -> Literal[PLANNER, ROUTER, FINALIZER]:  # type: ignore
    """When entering the supervisor subgraph, decide the entry point: plan, route, or finalize."""

    # if subtasks exists but not all are completed, router delegates to the next agent
    if state.subtasks:
        logger.debug("No need to plan as subtasks are already created.")
        return ROUTER

    # if there are no subtasks, come up with a plan
    logger.debug("Breaking down the query into subtasks.")
    return PLANNER


class SupervisorAgent:
    """Supervisor agent class."""

    model: IModel
    _name: str = SUPERVISOR
    members: list[str] = []
    plan_parser = PydanticOutputParser(pydantic_object=Plan)

    def __init__(
        self,
        models: dict[str, IModel | Embeddings],
        members: list[str],
        response_converter: IResponseConverter | None = None,
    ) -> None:
        self.model = cast(IModel, models[MAIN_MODEL_MINI_NAME])
        self.members = members

        self._router_chain = self._create_router_chain(self.model)
        self._planner_chain = self._create_planner_chain(self.model)
        self._graph = self._build_graph()

    def _get_members_str(self) -> str:
        return ", ".join(self.members)

    @property
    def name(self) -> str:
        """Agent name."""
        return self._name

    def agent_node(self) -> CompiledStateGraph:
        """Get Supervisor agent node function."""
        return self._graph

    def _create_router_chain(self, model: IModel) -> RunnableSequence:

        self.router_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ROUTER_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
                ("system", ROUTER_STEP_INSTRUCTIONS),
            ]
        ).partial(
            kyma_agent=KYMA_AGENT, kubernetes_agent=K8S_AGENT, common_agent=COMMON
        )
        return self.router_prompt | model.llm.with_structured_output(Route, method="function_calling")  # type: ignore

    def _route(self, state: SupervisorState) -> dict[str, Any]:
        """Router node. Routes the conversation to the next agent."""

        route = self._router_chain.invoke(state.messages)
        if route.next_agent != FINALIZER:
            # only send the subtask message to the dedicated agent
            return {
                "next": route.next_agent,
                "messages": [HumanMessage(content=route.task_description)],
            }

        # finalizer needs all the messages to generate the final response
        return {
            "next": FINALIZER,
        }

    def _create_planner_chain(self, model: IModel) -> RunnableSequence:
        self.planner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PLANNER_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
                ("system", PLANNER_STEP_INSTRUCTIONS),
            ]
        ).partial(
            kyma_agent=KYMA_AGENT, kubernetes_agent=K8S_AGENT, common_agent=COMMON
        )
        return self.planner_prompt | model.llm.with_structured_output(Plan, method="function_calling")  # type: ignore

    async def _invoke_planner(self, state: SupervisorState) -> Plan:
        """Invoke the planner with retry logic using tenacity."""

        filtered_messages = filter_messages_via_checks(
            state.messages,
            [
                is_human_message,
                is_system_message,
                is_finalizer_message,
                is_ai_message,
            ],
        )
        reduces_messages = filter_messages(filtered_messages)

        plan: Plan = await ainvoke_chain(
            self._planner_chain,
            {
                "messages": filter_valid_messages(reduces_messages),
            },
        )
        return plan

    async def _plan(self, state: SupervisorState) -> dict[str, Any]:
        """
        Breaks down the given user query into sub-tasks if the query is related to Kyma and K8s.
        If the query is general, it returns the response directly.
        """
        state.error = None

        try:
            plan = await self._invoke_planner(
                state,  # last message is the user query
            )

            # if the Planner failed to create any subtasks, raise an exception
            if not plan.subtasks:
                raise SubtasksMissingError(str(state.messages[-1].content))

            # return the plan with the subtasks to be dispatched by the Router
            subtasks_json = json.dumps(
                [subtask.model_dump(exclude={"status"}) for subtask in plan.subtasks],
                indent=2,
            )
            return create_node_output(
                message=AIMessage(
                    content=f"Here are the planned subtasks: \n{subtasks_json}",
                    name=PLANNER,
                ),  # This is needed to identify the planner
                next=ROUTER,
                subtasks=plan.subtasks,
            )
        except Exception:
            logger.exception("Error in planning")

            return create_node_output(
                message=AIMessage(
                    content="Unexpected error while processing the request. Please try again later.",
                    name=PLANNER,
                ),
                subtasks=[],  # empty subtask to make the companion response consistent
                next=END,
                error="Unexpected error while processing the request. Please try again later.",
            )

    def _build_graph(self) -> CompiledStateGraph:
        # Define a new graph.
        workflow = StateGraph(SupervisorState)

        # Define the nodes of the graph.
        workflow.add_node(PLANNER, self._plan)
        workflow.add_node(ROUTER, self._route)

        # Set the entrypoint: ENTRY --> (planner | router | finalizer)
        workflow.add_conditional_edges(
            START,
            decide_entry_point,
            {PLANNER: PLANNER, ROUTER: ROUTER},
        )

        # Define the edge: planner --> (router | END)
        workflow.add_conditional_edges(
            PLANNER,
            decide_route_or_exit,
            {ROUTER: ROUTER, END: END},
        )

        return workflow.compile()
