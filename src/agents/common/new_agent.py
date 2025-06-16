import json
from collections.abc import Sequence
from typing import (
    Annotated,
    TypedDict,
)

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools.base import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from services.k8s import IK8sClient
from utils.models.factory import IModel


class NewAgentState(TypedDict):
    """The state of the agent."""

    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]
    k8s_client: IK8sClient | None = None


class NewAgent:
    """New agent."""

    def __init__(
        self, name: str, system_prompt: str, tools: list[BaseTool], model: IModel
    ):
        self.name = name
        self.system_prompt = SystemMessage(system_prompt)
        self.tools = tools
        self.model = model.llm.bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.graph = self._build_graph()

    # Define our tool node
    def tool_node(self, state: NewAgentState):
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

    # Define the node that calls the model
    def call_model(
        self,
        state: NewAgentState,
        config: RunnableConfig,
    ):
        response = self.model.invoke([self.system_prompt] + state["messages"], config)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    # Define the conditional edge that determines whether to continue or not
    def should_continue(self, state: NewAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    def _build_graph(self):

        # Define a new graph
        workflow = StateGraph(NewAgentState)

        # Define the two nodes we will cycle between
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", ToolNode(tools=self.tools))

        # Set the entrypoint as `agent`
        # This means that this node is the first one called
        workflow.set_entry_point("agent")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            "agent",
            # Next, we pass in the function that will determine which node is called next.
            self.should_continue,
            # Finally we pass in a mapping.
            # The keys are strings, and the values are other nodes.
            # END is a special node marking that the graph should finish.
            # What will happen is we will call `should_continue`, and then the output of that
            # will be matched against the keys in this mapping.
            # Based on which one it matches, that node will then be called.
            {
                # If `tools`, then we call the tool node.
                "continue": "tools",
                # Otherwise we finish.
                "end": END,
            },
        )

        # We now add a normal edge from `tools` to `agent`.
        # This means that after `tools` is called, `agent` node is called next.
        workflow.add_edge("tools", "agent")

        # Now we can compile and visualize our graph
        return workflow.compile()
