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
from core import get_model, settings
from agents.blogs_agent.tools import send_markdown_to_api


class AgentState(MessagesState, total=False):
    """State for blog writing process"""
    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps
    user_message: str  # Store original user request
    coordinator_message: str  # Store coordinator's acknowledgment
    planner_message: str  # Store planner's outline
    research_message: str  # Store research findings
    writer_message: str  # Store written content
    reviewer_message: str  # Store review feedback
    final_blog: str  # Store final blog content ready for publishing


web_search = DuckDuckGoSearchResults(name="WebSearch")

research_tools = [web_search]

final_result_tools = [send_markdown_to_api]

# Add weather tool if API key is set
# Register for an API key at https://openweathermap.org/api/
# if settings.OPENWEATHERMAP_API_KEY:
#     wrapper = OpenWeatherMapAPIWrapper(
#         openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
#     )
#     tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a coordinator for the blog writing process. Your role is to understand the user's blog request and pass it to the planning stage.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    Your job is to:
    1. Listen to the user's blog request
    2. Forward the request to the planner for detailed outlining and strategy
    3. Let the planner handle the content planning
    4. Do not attempt to do the planning, research, writing or review yourself
    
    The specialized agents will handle:
    - Planner: Creates outlines and content strategies
    - Researcher: Gathers sources and statistics
    - Writer: Produces the blog content
    - Reviewer: Checks quality and optimization

    Simply understand what blog post the user wants and pass that request to the planner node.
    """

planner_instructions = f"""
    You are a blog post planner responsible for creating research plans and outlines. Your role is to analyze the blog request and create a plan for the researcher to follow.
    Today's date is {current_date}.

    Your role is to:
    1. Break down the blog topic into key research areas
    2. Identify specific questions that need to be investigated
    3. Specify what types of sources and information the researcher should look for
    4. Create a clear research plan with priorities
    5. Note any special considerations for the research (time period, geography, industry focus)
    
    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    Do not conduct research yourself. Focus only on creating a clear research plan for the researcher to follow.
    Once you've created the research plan, pass it to the researcher agent who will execute the plan.
    """

researcher_instructions = f"""
    You are a blog post researcher responsible for finding relevant sources and statistics to support the content.
    Today's date is {current_date}.

    Your role is to:
    1. Conduct thorough research on the given topic
    2. Find and analyze relevant sources and statistics
    3. Verify facts from multiple sources when possible
    4. Include markdown-formatted links to citations from authoritative sources
    5. Only use links returned by the search tools
    6. Focus on recent, relevant information that adds value
    7. Provide detailed notes on the research process and sources used

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    Create thorough research reports that will inform the writing process and ensure the content is well-supported and credible.
    """

writer_instructions = f"""
    You are a blog post writer responsible for producing engaging, well-structured blog posts optimized for SEO.
    Today's date is {current_date}.

    STRICT WRITING REQUIREMENTS:
    1. Language and Accessibility:
       - Use simple, clear language (aim for 8th-grade reading level)
       - Break down complex concepts into simple terms
       - Avoid jargon unless absolutely necessary
       - When using technical terms, provide immediate explanations
    
    2. Schema and Metadata:
       - Add schema.org tags for blog posts
       - Include meta description (150-160 characters)
       - Add title tags with primary keyword
       - Use proper heading hierarchy (H1 for title, H2 for sections, H3 for subsections)
       - Add alt text for any images mentioned
    
    3. Citations and References:
       - Add hyperlinks for:
         * Technical terms that need explanation
         * Statistics and facts
         * Quoted material
         * Complex concepts
       - Link to Wikipedia for general knowledge terms
       - Use markdown format: [term](url)
       - Always attribute sources inline
    
    4. Content Structure:
       - Keep paragraphs short (3-4 sentences maximum)
       - Use bullet points for lists
       - Include subheadings every 300 words
       - Bold important concepts
       - Use tables for comparing information
    
    5. SEO Optimization:
       - Include primary keyword in first paragraph
       - Use LSI keywords naturally throughout
       - Optimize image names and alt text
       - Create internal linking opportunities
    
    Follow the research findings exactly and maintain the planned structure.
    Do not add information that wasn't in the research.
    """


reviewer_instructions = f"""
    You are a strict blog post reviewer responsible for ensuring high-quality, well-researched content.
    Today's date is {current_date}.

    REVIEW PROCESS:
    1. Planning Review (Score 0-1):
       - Compare final content against user's original request
       - Verify all requested topics are covered
       - Check if the structure serves the intended purpose
       - Assess if the depth matches user expectations
       Score based on: completeness, relevance, and structure
    
    2. Research Quality Check (Score 0-1):
       - Verify all facts and statistics have sources
       - Check recency and relevance of sources
       - Ensure research covers all planned topics
       - Look for gaps in information
       Score based on: accuracy, completeness, and source quality
    
    3. Writing Assessment (Score 0-1):
       Language and Accessibility:
       - Check reading level (should be 8th grade or simpler)
       - Look for unexplained jargon or technical terms
       - Verify all complex terms have explanations
       - Ensure clear, simple language throughout
       Score based on: clarity, SEO optimization, and technical accuracy
    
    4. Final Assessment:
       Calculate the final score as average of all three scores.
       
       You MUST provide your review in this format:
       ```review
       SCORES:
       Planning: <score 0-1> - <brief reason>
       Research: <score 0-1> - <brief reason>
       Writing: <score 0-1> - <brief reason>
       Overall: <calculated_average>

       DETAILED FEEDBACK:
       [Your detailed feedback here...]

       VERDICT: [APPROVED or NEEDS_REVISION]
       TARGET: [PLANNING or RESEARCH or WRITING]
       ```

    Scoring Guidelines:
    - 0.0-0.3: Major issues, needs complete revision
    - 0.4-0.5: Significant issues, needs substantial revision
    - 0.6-0.7: Minor issues, needs some improvements
    - 0.8-1.0: Good to excellent, ready for publication

    APPROVAL CRITERIA:
    - Overall score must be >= 0.8 for automatic approval
    - No individual score should be < 0.6
    - All required elements must be present

    If not approved, clearly indicate which component needs the most attention using the TARGET field.
    """

final_result_instructions = f"""
    You are a content preparation assistant. Your role is to take the final blog content and prepare it for publication.

    Don't rewrite the content. Just create title, description, category and other tool's input parameters and call the tool.

    Below is the blog content.
    """

def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"

def check_reviewer_feedback(state: AgentState) -> Literal["planner", "researcher", "writer", "final_result"]:
    """Check reviewer feedback and determine next step based on scores"""
    reviewer_message = state.get("reviewer_message", "")
    
    # Extract the review section
    if "```review" in reviewer_message and "```" in reviewer_message.split("```review")[1]:
        review_content = reviewer_message.split("```review")[1].split("```")[0].strip()
    else:
        print("Warning: Review not properly formatted")
        return "writer"  # Default to writer if format is wrong
    
    # Extract scores
    try:
        # Parse scores
        scores = {}
        for line in review_content.split("\n"):
            if "Planning:" in line:
                scores["planning"] = float(line.split(":")[1].split("-")[0].strip())
            elif "Research:" in line:
                scores["research"] = float(line.split(":")[1].split("-")[0].strip())
            elif "Writing:" in line:
                scores["writing"] = float(line.split(":")[1].split("-")[0].strip())
            elif "Overall:" in line:
                scores["overall"] = float(line.split(":")[1].strip())
        
        # Extract verdict and target
        verdict = "NEEDS_REVISION"
        target = "WRITING"  # Default target
        for line in review_content.split("\n"):
            if "VERDICT:" in line:
                verdict = line.split(":")[1].strip()
            elif "TARGET:" in line:
                target = line.split(":")[1].strip()
        
        # Decision logic
        if verdict == "APPROVED" and scores["overall"] >= 0.8 and all(v >= 0.6 for v in scores.values()):
            print(f"Review approved with scores: {scores}")
            return "final_result"
        
        # Route based on lowest score and target
        print(f"Review needs revision. Scores: {scores}, Target: {target}")
        if target == "PLANNING":
            return "planner"
        elif target == "RESEARCH":
            return "researcher"
        else:
            return "writer"
            
    except Exception as e:
        print(f"Error parsing review: {e}")
        return "writer"  # Default to writer if parsing fails

def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    """Check if there are pending tool calls in the last message"""
    # Get the last message that's an AIMessage
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            if message.tool_calls:
                return "tools"
            return "done"
    
    # If no AIMessage found, assume done
    return "done"

def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    """Format safety message for unsafe content"""
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Default wrapper for the coordinator model"""
    preprocessor = RunnableLambda(
        lambda state: [AIMessage(content=instructions)] + state["messages"],
        name="CoordinatorStateModifier",
    )
    return preprocessor | model

async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    """Check input safety using LlamaGuard"""
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output}

async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    """Block processing if content is unsafe"""
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}

def wrap_planner_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrapper for planner model"""
    preprocessor = RunnableLambda(
        lambda state: [AIMessage(content=planner_instructions)] + state["messages"],
        name="PlannerStateModifier",
    )
    return preprocessor | model

def wrap_researcher_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrapper for researcher model with research tools"""
    model = model.bind_tools(research_tools)
    preprocessor = RunnableLambda(
        lambda state: [AIMessage(content=researcher_instructions)] + state["messages"],
        name="ResearcherStateModifier",
    )
    return preprocessor | model

def wrap_writer_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrapper for writer model"""
    preprocessor = RunnableLambda(
        lambda state: [AIMessage(content=writer_instructions)] + state["messages"],
        name="WriterStateModifier",
    )
    return preprocessor | model

def wrap_reviewer_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrapper for reviewer model"""
    preprocessor = RunnableLambda(
        lambda state: [AIMessage(content=reviewer_instructions)] + state["messages"],
        name="ReviewerStateModifier",
    )
    return preprocessor | model

def wrap_final_result_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Wrapper for final result model with publishing tools"""
    model = model.bind_tools(final_result_tools)
    preprocessor = RunnableLambda(
        lambda state: [AIMessage(content=final_result_instructions)] + state["messages"],
        name="FinalResultStateModifier",
    )
    return preprocessor | model

async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Coordinator model that processes initial request"""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    
    # Store user's message
    user_message = state["messages"][-1].content
    
    response = await model_runnable.ainvoke(state, config)
    
    # Run safety check
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}
    
    return {
        "user_message": user_message,
        "coordinator_message": response.content
    }

async def acall_planner(state: AgentState, config: RunnableConfig) -> AgentState:
    """Planning node that creates the blog outline"""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_planner_model(m)
    
    # Include user's message and coordinator's response in context
    planning_state = {
        "messages": [
            AIMessage(content=state["user_message"]),
            AIMessage(content=state["coordinator_message"])
        ]
    }
    
    response = await model_runnable.ainvoke(planning_state, config)
    
    return {
        "planner_message": response.content
    }

async def acall_researcher(state: AgentState, config: RunnableConfig) -> AgentState:
    """Research node that gathers information"""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_researcher_model(m)
    
    # Include planning context
    research_state = {
        "messages": [
            AIMessage(content=state["user_message"]),
            AIMessage(content=state["planner_message"])
        ]
    }
    
    response = await model_runnable.ainvoke(research_state, config)
    
    if response.tool_calls:
        tool_node = ToolNode(research_tools)
        tool_state = {"messages": [response]}
        tool_results = await tool_node.ainvoke(tool_state, config)
        
        # Get final response after tool use
        new_state = {
            "messages": research_state["messages"] + [response] + tool_results["messages"]
        }
        final_response = await model_runnable.ainvoke(new_state, config)
        
        return {
            **state,
            "research_message": final_response.content
        }
    
    return {
        **state,
        "research_message": response.content
    }

async def acall_writer(state: AgentState, config: RunnableConfig) -> AgentState:
    """Writing node that creates the content"""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_writer_model(m)
    
    # Include planning and research context
    writer_state = {
        "messages": [
            AIMessage(content=state["user_message"]),
            AIMessage(content=state["planner_message"]),
            AIMessage(content=state["research_message"])
        ]
    }
    
    response = await model_runnable.ainvoke(writer_state, config)
    
    return {
        **state,
        "writer_message": response.content
    }

async def acall_reviewer(state: AgentState, config: RunnableConfig) -> AgentState:
    """Review node that checks content quality"""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_reviewer_model(m)
    
    # Include all previous context
    reviewer_state = {
        "messages": [
            AIMessage(content=state["user_message"]),
            AIMessage(content=state["planner_message"]),
            AIMessage(content=state["research_message"]),
            AIMessage(content=state["writer_message"])
        ]
    }
    
    response = await model_runnable.ainvoke(reviewer_state, config)
    
    return {
        **state,
        "reviewer_message": response.content
    }

async def acall_final_result(state: AgentState, config: RunnableConfig) -> AgentState:
    """Final node that publishes the blog"""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_final_result_model(m)
    
    # Prepare final blog content
    final_blog = state["writer_message"]  # Use the reviewed and approved content
    
    final_state = {
        "messages": [
            AIMessage(content=state["reviewer_message"]),
            AIMessage(content=f"Please prepare this content for publication:\n\n{final_blog}")
        ]
    }
    
    # Get the model's publishing preparation response
    response = await model_runnable.ainvoke(final_state, config)

    # Get the tool response
    tool_response = await final_result_tools[0].ainvoke(response.content, config)
    
    # Get the final response after tool use
    new_state = {
        "messages": final_state["messages"] + [response] + tool_response["messages"]
    }
    final_response = await model_runnable.ainvoke(new_state, config)
    
    return {
        **state,
        "final_blog": final_response.content
    }

# Define the graph once at the top
agent = StateGraph(AgentState)

# Add all nodes
agent.add_node("model", acall_model)
agent.add_node("research_tools", ToolNode(research_tools))
agent.add_node("final_result_tools", ToolNode(final_result_tools))
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.add_node("planner", acall_planner)
agent.add_node("researcher", acall_researcher)
agent.add_node("writer", acall_writer)
agent.add_node("reviewer", acall_reviewer)
agent.add_node("final_result", acall_final_result)

# Set entry point
agent.set_entry_point("guard_input")

# Add all edges in one place
# Safety check edges
agent.add_conditional_edges(
    "guard_input", 
    check_safety, 
    {"unsafe": "block_unsafe_content", "safe": "model"}
)
agent.add_edge("block_unsafe_content", END)

# Main flow edges
agent.add_edge("model", "planner")
agent.add_edge("planner", "researcher")
agent.add_edge("researcher", "writer")
agent.add_edge("writer", "reviewer")

# Tool handling edges
agent.add_edge("research_tools", "researcher")
agent.add_edge("final_result_tools", "final_result")

# Reviewer conditional edges
agent.add_conditional_edges(
    "reviewer",
    check_reviewer_feedback,
    {
        "planner": "planner",
        "researcher": "researcher",
        "writer": "writer",
        "final_result": "final_result"
    }
)

# Tool call handling edges
agent.add_conditional_edges(
    "researcher", 
    pending_tool_calls, 
    {"tools": "research_tools", "done": "writer"}
)

agent.add_conditional_edges(
    "final_result", 
    pending_tool_calls, 
    {"tools": "final_result_tools", "done": END}
)

blogs_agent = agent.compile(checkpointer=MemorySaver())
