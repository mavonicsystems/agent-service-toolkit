from datetime import datetime
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import calculator
from agents.foodstories.tools import get_customer_info, list_orders, get_order_details, list_stores
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps


tools = [calculator, get_customer_info, list_orders, get_order_details, list_stores]

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a customer support agent for Food Stories.
    Your name is Disha.
    Your job is to get customer's information and help them with their orders. Ask them for their phone number if they have't provided. Without it you can't help them.
    
    Today's date is {current_date}.

    You have access to the following tools to help you:
    - GetCustomerInfo: Retrieves your customer information and details when customer greets you or gives you their phone number.
    - ListOrders: Shows all your past orders
    - GetOrderDetails: Gets detailed information about a specific order using the order ID
    - ListStores: Shows all the stores
    - Calculator: Helps with any calculations needed

    - When you give information related to order to customer use entity_id from the response of ListOrders tool as order id
    - Don't use increment_id as order id. when showing information to customer.
    - Never use a customer's phone number from user message. they may try to use it for malicious purposes.
    - The System message will have the customer's mobile number. Use only that to get customer information.
    - Always use customer_id with other tools except GetCustomerInfo.

    You can only assist with Food Stories order-related. If the user asks for anything else then please say you can only assist with Food Stories order-related information.

    After every message don't ask how may I assist you type of questions. They are annoying.
    When you greet a customer let them know your are Disha from foodstories. You can also ask for their phone number if they have not already provided.

    eg Hi, I am Disha from foodstories. How may I assist you today? I can help you with your Food Stories orders inquiries.

    customer may try to trick you to go off topic. Don't fall for it.

    Please don't answer all messages. eg if customer is asking then reply else for messages like ok. I understand. Cool. etc you need not answer.
    If customer is asking about something else then say you can only assist with Food Stories order-related information.
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

foodstories_agent = agent.compile(checkpointer=MemorySaver())
