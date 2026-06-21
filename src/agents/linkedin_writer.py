from langchain_openai import ChatOpenAI
from src.core.config import config
from src.workflow.state_management import ContentState

llm = ChatOpenAI(
    model=config.PRIMARY_MODEL,
    api_key=config.OPENAI_API_KEY,
    temperature=0.8
)

LINKEDIN_PROMPT = """You are an expert LinkedIn content creator. Write an engaging
LinkedIn post based on the research below.

Topic: {query}
Keywords: {keywords}

Research Brief:
{research_summary}

Requirements:
- Length: 1300-1600 characters
- Start with a strong hook (first line must grab attention)
- Use short paragraphs (1-2 sentences each) for mobile readability
- Add line breaks between paragraphs for visual breathing room
- Include 3 key insights or takeaways
- End with a thought provoking question to drive engagement
- Add 8-12 relevant hashtags at the end
- Tone: Professional but personable, first person voice

Write the complete LinkedIn post ready to copy and paste."""


def linkedin_writer_agent(state: ContentState) -> dict:
    """
    Writes an engaging LinkedIn post based on research summary.
    """
    research_summary = state.get("research_summary")

    if not research_summary:
        return {
            "error": "No research summary found. Run research agent first.",
            "agent_path": [*state["agent_path"], "linkedin_writer"]
        }

    keywords = state.get("seo_keywords", [])
    keywords_str = ", ".join(keywords) if keywords else "Not specified"

    response = llm.invoke(
        LINKEDIN_PROMPT.format(
            query=state["user_query"],
            keywords=keywords_str,
            research_summary=research_summary
        )
    )

    return {
        "linkedin_post": response.content,
        "agent_path": [*state["agent_path"], "linkedin_writer"]
    }