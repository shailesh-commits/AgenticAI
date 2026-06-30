#!/usr/bin/env python
# coding: utf-8

# In[19]:


get_ipython().system('pip install -q U langgraph langchain-openai tool python-dotenv')


# In[1]:


from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import os
import dotenv

# 1. Setup Environment Variables
# In a real notebook, use getpass or an .env file
dotenv.load_dotenv()
#openai_api_key = os.environ["OPENAI_API_KEY"]

# ==========================================
# 2. DEFINE THE STATE & TOOLS
# ==========================================

class LabState(TypedDict):
    """The state passed between nodes in our LangGraph workflow."""
    # add_messages appends new messages to history instead of overwriting them
    messages: Annotated[list[BaseMessage], add_messages]
    leave_balance_checked: bool
    is_approved_for_next_stage: bool

@tool
def check_leave_balance(employee_id: str, leave_type: str) -> str:
    """Queries the HR database to get the remaining leave days for an employee."""
    # Mock database logic for training clarity
    mock_db = {
        "EMP123": {"casual": 5, "medical": 12, "annual": 2},
        "EMP999": {"casual": 0, "medical": 1, "annual": 0}
    }

    emp_record = mock_db.get(employee_id.upper())
    if not emp_record:
        return f"Employee ID {employee_id} not found in system."

    balance = emp_record.get(leave_type.lower(), 0)
    return f"Employee {employee_id} has {balance} days of {leave_type} leave remaining."

# Bundle tools together
tools = [check_leave_balance]
tool_node = ToolNode(tools)

# ==========================================
# 3. DEFINE THE AGENT LOGIC & NODES
# ==========================================

# Initialize the OpenAI model with tools bound to it
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def intake_agent(state: LabState) -> dict:
    """The primary agent node analyzing the input and deciding to call tools or respond."""
    system_prompt = (
        "You are an Intelligent HR Intake Assistant. Your job is to extract: \n"
        "1. Employee ID, 2. Leave Type (casual/medical/annual), 3. Duration.\n"
        "Always use the `check_leave_balance` tool immediately if you find an Employee ID and Leave Type. "
        "If information is missing, politely ask the user to provide it."
    )

    messages = [AIMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

# ==========================================
# 4. DEFINE THE ROUTING LOGIC
# ==========================================

def route_after_agent(state: LabState) -> Literal["tools", "evaluate_eligibility"]:
    """Determines if the graph needs to execute tools or move to policy checking."""
    last_message = state["messages"][-1]

    # If the LLM decided to call a tool, route to the tool node
    if last_message.tool_calls:
        return "tools"

    # Otherwise, evaluate what to do next based on tool output history
    return "evaluate_eligibility"

def evaluate_eligibility(state: LabState) -> dict:
    """Analyzes tool output in history to check if the request can move to managers."""
    # Simple evaluation node to update state variables based on conversation flow
    messages = state["messages"]

    # Check if a tool execution message is in the history
    has_checked_balance = any(msg.type == "tool" for msg in messages)

    # For training simplicity, if a tool was run successfully, flag it as ready for next stage
    if has_checked_balance:
        return {
            "leave_balance_checked": True, 
            "is_approved_for_next_stage": True,
            "messages": [AIMessage(content="System Log: Balance validated. Ready for Manager Review Stage.")]
        }

    return {"leave_balance_checked": False, "is_approved_for_next_stage": False}

def route_final(state: LabState) -> Literal["__end__", "intake_agent"]:
    """Determines whether to end the graph or cycle back for more user input."""
    if state.get("is_approved_for_next_stage"):
        return END
    # If not ready, cycle back to agent to ask user clarifying questions
    return "intake_agent"

# ==========================================
# 5. COMPILE THE GRAPH
# ==========================================

workflow = StateGraph(LabState)

# Add Nodes
workflow.add_node("intake_agent", intake_agent)
workflow.add_node("tools", tool_node)
workflow.add_node("evaluate_eligibility", evaluate_eligibility)

# Set Entry Point
workflow.add_edge(START, "intake_agent")

# Add Conditional Edges
workflow.add_conditional_edges(
    "intake_agent",
    route_after_agent,
)

workflow.add_edge("tools", "intake_agent")

workflow.add_conditional_edges(
    "evaluate_eligibility",
    route_final
)

# Compile Graph
app = workflow.compile()


# In[2]:


# The agent will realize it's missing an Employee ID and ask for it
inputs = {"messages": [HumanMessage(content="Hey, I want to take a casual leave tomorrow.")]}
config = {"configurable": {"thread_id": "1"}}

events = app.stream(inputs, config, stream_mode="values")
for event in events:
    event["messages"][-1].pretty_print()


# In[3]:


# The agent has all the information, triggers the tool, and completes the workflow
inputs = {"messages": [HumanMessage(content="Hi, my employee ID is EMP123. I'm feeling sick and need 2 days of medical leave.")]}
config = {"configurable": {"thread_id": "2"}}

events = app.stream(inputs, config, stream_mode="values")
for event in events:
    print(f"\n--- Current Graph Node Executed ---")
    event["messages"][-1].pretty_print()


# In[ ]:




