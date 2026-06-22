from typing import TypedDict, Annotated, Optional
import operator


class ContentState(TypedDict):
    """
    Shared state that flows through all agents in the graph.
    Each agent reads from this and writes updates back to it.
    """
    # Conversation
    messages: Annotated[list, operator.add]
    user_query: str
    conversation_id: str

    # Routing
    intent: Optional[str]        # "research" | "blog" | "linkedin" | "image" | "blog_linkedin" | "full_content"
    agent_path: list[str]        # tracks which agents ran

    # Research
    research_results: Optional[list[dict]]
    research_summary: Optional[str]

    # Content outputs
    blog_post: Optional[str]
    linkedin_post: Optional[str]
    image_prompt: Optional[str]
    image_url: Optional[str]
    image_local_path: Optional[str] 

    # Quality
    quality_score: Optional[float]
    seo_keywords: Optional[list[str]]

    # Errors
    error: Optional[str]
    fallback_used: Optional[str]


def create_initial_state(user_query: str, conversation_id: str) -> ContentState:
    """Factory function — creates a fresh state for every new user request."""
    return ContentState(
        messages=[{"role": "user", "content": user_query}],
        user_query=user_query,
        conversation_id=conversation_id,
        intent=None,
        agent_path=[],
        research_results=None,
        research_summary=None,
        blog_post=None,
        linkedin_post=None,
        image_prompt=None,
        image_url=None,
        image_local_path=None,
        quality_score=None,
        seo_keywords=None,
        error=None,
        fallback_used=None,
    )