from langchain_openai import ChatOpenAI
from src.core.config import config
from src.workflow.state_management import ContentState

llm = ChatOpenAI(
    model=config.PRIMARY_MODEL,
    api_key=config.OPENAI_API_KEY,
    temperature=0.7
)

BLOG_PROMPT = """You are an expert SEO blog writer. Write a comprehensive, 
engaging blog post based on the research below.

Topic: {query}
Target Keywords: {keywords}

Research Brief:
{research_summary}

Requirements:
- Length: 1200-1500 words
- Structure: H1 title, H2 sections, H3 subsections where needed
- Include target keywords naturally (1-2% density)
- Write a compelling introduction that hooks the reader
- Include practical insights and actionable takeaways
- End with a strong conclusion and call to action
- Meta description: Write one at the end (150-160 characters)
- Tone: Professional yet conversational

Write the complete blog post in markdown format."""


def blog_writer_agent(state: ContentState) -> dict:
    """
    Writes an SEO optimized blog post based on research summary.
    """
    # Prefer content_brief (structured) over raw research_summary
    research_summary = state.get("content_brief") or state.get("research_summary")

    if not research_summary:
        return {
            "error": "No research summary found. Run research agent first.",
            "agent_path": [*state["agent_path"], "blog_writer"]
        }

    keywords = state.get("seo_keywords", [])
    keywords_str = ", ".join(keywords) if keywords else "Not specified"

    response = llm.invoke(
        BLOG_PROMPT.format(
            query=state["user_query"],
            keywords=keywords_str,
            research_summary=research_summary
        )
    )

    return {
        "blog_post": response.content,
        "agent_path": [*state["agent_path"], "blog_writer"]
    }