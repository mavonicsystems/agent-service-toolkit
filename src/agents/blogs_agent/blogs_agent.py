from datetime import datetime
from typing import Literal, Dict, List, TypedDict, Any
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from operator import add
from functools import partial

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
from core import get_model, settings
from agents.blogs_agent.tools import send_markdown_to_api


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps


class BlogSection(BaseModel):
    """Schema for a blog section"""
    title: str = Field(description="Title of the section")
    key_points: List[str] = Field(description="Key points to cover in this section")

class BlogPlan(BaseModel):
    """Schema for the blog plan"""
    title: str = Field(description="Main title of the blog")
    sections: List[BlogSection] = Field(description="List of sections to write")

class BlogState(AgentState, total=False):
    """State for blog writing agent"""
    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps
    plan: BlogPlan
    section_contents: Dict[str, str]
    final_blog: str


web_search = DuckDuckGoSearchResults(name="WebSearch")
tools = [web_search, calculator, send_markdown_to_api]

# Add weather tool if API key is set
# Register for an API key at https://openweathermap.org/api/
if settings.OPENWEATHERMAP_API_KEY:
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful research assistant with the ability to search the web, use tools, and publish blog posts.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
      so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".
    - After writing a blog post, use the send_markdown_to_api tool to publish it.
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


async def plan_blog(state: BlogState, config: RunnableConfig) -> BlogState:
    """Plan the blog sections"""
    planning_prompt = """
    Based on the user's request, create a detailed blog post plan.
    Break it down into logical sections that can be written independently.
    
    Format the output as JSON matching this schema:
    {
        "title": "Main blog title",
        "sections": [
            {
                "title": "Section title",
                "key_points": ["point 1", "point 2", "point 3"]
            }
        ]
    }
    
    Requirements:
    - Create 3-5 sections
    - Each section should have 2-4 key points
    - Make sections logically connected but independently writable
    """
    
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    parser = PydanticOutputParser(pydantic_object=BlogPlan)
    
    # First get the model's response
    response = await m.ainvoke(
        [
            SystemMessage(content=planning_prompt),
            SystemMessage(content=f"Output format instructions: {parser.get_format_instructions()}"),
        ] + state["messages"],
        config
    )
    
    # Then parse the response into our BlogPlan structure
    try:
        plan = parser.parse(response.content)
        return {"plan": plan}
    except Exception as e:
        # If parsing fails, return an error message
        return {
            "messages": [
                AIMessage(content=f"Failed to create blog plan: {str(e)}\nResponse was: {response.content}")
            ]
        }


async def write_sections(state: BlogState, config: RunnableConfig) -> BlogState:
    """Write all blog sections sequentially"""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    sections_content = {}
    
    # First, create a context of the overall blog structure
    blog_context = f"""
    You are writing a blog post titled: "{state['plan'].title}"
    
    The blog has the following sections:
    {chr(10).join(f'- {section.title}' for section in state['plan'].sections)}
    """
    
    for i, section in enumerate(state["plan"].sections):
        # Add context about previously written sections
        previous_sections = ""
        if i > 0:
            previous_sections = "\nPreviously written sections:\n"
            for prev_section in state["plan"].sections[:i]:
                if prev_section.title in sections_content:
                    previous_sections += f"\n## {prev_section.title}\n{sections_content[prev_section.title]}\n"
        
        writing_prompt = f"""
        {blog_context}
        
        {previous_sections}
        
        Now, write the following section:
        
        SECTION TITLE: {section.title}
        
        KEY POINTS TO COVER:
        {chr(10).join(f'- {point}' for point in section.key_points)}
        
        Write in a clear, engaging style. Use markdown formatting.
        Make sure this section flows naturally from previous sections and maintains the blog's coherence.
        Don't repeat information that was already covered in previous sections.
        Focus on providing unique value in this section while maintaining the overall narrative.
        """
        
        response = await m.ainvoke(
            [SystemMessage(content=writing_prompt)] + state["messages"],
            config
        )
        sections_content[section.title] = response.content
    
    return {"section_contents": sections_content}


async def combine_sections(state: BlogState, config: RunnableConfig) -> BlogState:
    """Combine all sections into final blog post"""
    plan: BlogPlan = state["plan"]
    sections: Dict[str, str] = state["section_contents"]
    
    # Start with just the content, no title (will be handled by the publishing tool)
    final_blog = ""
    for section in plan.sections:
        if section.title in sections:
            # Don't include the section title in the content since it's already in the markdown
            content = sections[section.title]
            # Remove any "## Section Title" that might be at the start of the content
            content = content.replace(f"## {section.title}\n", "").replace(f"## {section.title}\r\n", "")
            final_blog += f"\n## {section.title}\n\n{content}\n"
    
    return {"final_blog": final_blog, "messages": [AIMessage(content=final_blog)]}


async def publish_blog(state: BlogState, config: RunnableConfig) -> BlogState:
    """Publish the blog using the API tool"""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    
    # Calculate read time directly
    word_count = len(state["final_blog"].split())
    read_time = max(1, round(word_count / 238))  # minimum 1 minute, using standard 238 wpm
    read_time_str = f"{read_time} minutes"
    
    # First, generate blog metadata
    metadata_prompt = f"""
    Based on the blog content, generate:
    1. A concise but engaging description (2-3 sentences)
    2. The most appropriate category for this blog from these options ONLY:
       - Technology
       - Business
       - Science
       - Health
       - Lifestyle
       - Education
       - Opinion
    
    You MUST choose one of these exact category names, no variations allowed.
    Format your response as a valid JSON object with no additional text:
    {{
        "description": "your description here",
        "category": "category_name"
    }}
    """
    
    metadata_response = await m.ainvoke(
        [SystemMessage(content=metadata_prompt)],
        config
    )
    
    try:
        import json
        # Extract JSON from markdown code block if present
        content = metadata_response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        metadata = json.loads(content)
        # Validate category is one of the allowed values
        allowed_categories = {"Technology", "Business", "Science", "Health", "Lifestyle", "Education", "Opinion"}
        if metadata["category"] not in allowed_categories:
            raise ValueError(f"Invalid category: {metadata['category']}")
    except Exception as e:
        print(f"Failed to parse or validate metadata: {e}\nResponse was: {metadata_response.content}")
        metadata = {
            "description": "An informative blog post about " + state['plan'].title,
            "category": "Education"  # Default fallback
        }
    # Create the tool call prompt with explicit JSON structure
    tool_call_data = {
        "title": state['plan'].title,
        "description": metadata['description'],
        "content": state['final_blog'],
        "read_time": read_time_str,
        "category": metadata['category']
    }
    
    publish_prompt = f"""
    Use the send_markdown_to_api tool to publish this blog.
    You MUST use EXACTLY these parameters, do not modify any values:
    {json.dumps(tool_call_data, indent=2)}

    Make the tool call with these exact parameters, no modifications allowed.
    Do not add any additional text before or after the tool call.
    """
    
    # Create a proper state with messages for the model
    publish_state = {
        "messages": [
            SystemMessage(content=instructions),
            SystemMessage(content=publish_prompt)
        ]
    }
    
    response = await model_runnable.ainvoke(publish_state, config)
    
    # If the model made a tool call, we need to process it
    if response.tool_calls:
        tool_node = ToolNode(tools)
        tool_state = {"messages": [response]}
        tool_result = await tool_node.ainvoke(tool_state, config)
        
        # Add both the tool call and its result to the messages
        return {"messages": state["messages"] + [response] + tool_result["messages"]}
    
    # If no tool call was made, return an error message
    return {
        "messages": state["messages"] + [
            AIMessage(content=f"Failed to publish the blog. Intended parameters were: {json.dumps(tool_call_data, indent=2)}")
        ]
    }


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"

# Define the graph
agent = StateGraph(BlogState)

# Add nodes
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.add_node("planning", plan_blog)
agent.add_node("write", write_sections)
agent.add_node("combine", combine_sections)
agent.add_node("publish", publish_blog)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))

# Set up the flow
agent.set_entry_point("guard_input")

# Add edges
agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "planning"}
)
agent.add_edge("block_unsafe_content", END)
agent.add_edge("planning", "write")
agent.add_edge("write", "combine")
agent.add_edge("combine", "publish")
agent.add_edge("publish", END)

# Tool handling edges
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

blogs_agent = agent.compile(checkpointer=MemorySaver())
