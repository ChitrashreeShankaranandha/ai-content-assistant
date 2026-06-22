from langgraph.graph import StateGraph, END
from src.workflow.state_management import ContentState
from src.agents.query_handler import query_handler_agent
from src.agents.research_agent import research_agent
from src.agents.content_strategist import content_strategist_agent
from src.agents.blog_writer import blog_writer_agent
from src.agents.linkedin_writer import linkedin_writer_agent
from src.agents.image_generator import image_generator_agent


def route_by_intent(state: ContentState) -> str:
    """
    Conditional edge function — reads the intent set by query_handler
    and returns the name of the next node to execute.
    """
    intent = state.get("intent", "research")

    if intent == "research":
        return "research"
    elif intent == "blog":
        return "research"  # blog needs research first
    elif intent == "linkedin":
        return "research"  # linkedin needs research first
    elif intent == "image":
        return "image_generator"
    elif intent == "full_content":
        return "research"  # full content starts with research
    else:
        return "research"


def route_after_research(state: ContentState) -> str:
    """After research, decide where to go based on original intent."""
    intent = state.get("intent", "research")

    if intent == "research":
        return END  # research only — we're done
    else:
        return "content_strategist"  # otherwise structure the research


def route_after_strategist(state: ContentState) -> str:
    """After strategist, route to the right content generator."""
    intent = state.get("intent", "blog")

    if intent == "blog":
        return "blog_writer"
    elif intent == "linkedin":
        return "linkedin_writer"
    elif intent == "blog_linkedin":
        return "blog_writer"  # blog first, then linkedin (no image)
    elif intent == "full_content":
        return "blog_writer"  # blog first, then linkedin, then image
    else:
        return END
    

def route_after_blog(state: ContentState) -> str:
    """After blog, continue to linkedin for blog_linkedin or full_content."""
    intent = state.get("intent", "blog")
    if intent in ["blog_linkedin", "full_content"]:
        return "linkedin_writer"
    return END


def route_after_linkedin(state: ContentState) -> str:
    """After linkedin, only full_content continues to image generator."""
    intent = state.get("intent", "linkedin")
    if intent == "full_content":
        return "image_generator"
    return END


def build_workflow():
    """
    Builds and compiles the LangGraph workflow.
    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(ContentState)

    # Add all agents as nodes
    graph.add_node("query_handler", query_handler_agent)
    graph.add_node("research", research_agent)
    graph.add_node("content_strategist", content_strategist_agent)
    graph.add_node("blog_writer", blog_writer_agent)
    graph.add_node("linkedin_writer", linkedin_writer_agent)
    graph.add_node("image_generator", image_generator_agent)

    # Entry point
    graph.set_entry_point("query_handler")

    # Conditional routing from query_handler
    graph.add_conditional_edges(
        "query_handler",
        route_by_intent,
        {
            "research": "research",
            "image_generator": "image_generator"
        }
    )

    # After research, decide whether to continue
    graph.add_conditional_edges(
        "research",
        route_after_research,
        {
            "content_strategist": "content_strategist",
            END: END
        }
    )

    # After strategist, route to the right writer
    graph.add_conditional_edges(
        "content_strategist",
        route_after_strategist,
        {
            "blog_writer": "blog_writer",
            "linkedin_writer": "linkedin_writer",
            END: END
        }
    )

    # After blog, optionally continue to linkedin
    graph.add_conditional_edges(
        "blog_writer",
        route_after_blog,
        {
            "linkedin_writer": "linkedin_writer",
            END: END
        }
    )

    # After linkedin, optionally continue to image
    graph.add_conditional_edges(
        "linkedin_writer",
        route_after_linkedin,
        {
            "image_generator": "image_generator",
            END: END
        }
    )

    # Image generator is always the last step
    graph.add_edge("image_generator", END)

    return graph.compile()


# Module level singleton — build once, reuse everywhere
workflow = build_workflow()