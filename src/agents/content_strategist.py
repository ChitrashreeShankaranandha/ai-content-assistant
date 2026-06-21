from langchain_openai import ChatOpenAI
from src.core.config import config
from src.workflow.state_management import ContentState

llm = ChatOpenAI(
    model=config.PRIMARY_MODEL,
    api_key=config.OPENAI_API_KEY,
    temperature=0.5
)

STRATEGIST_PROMPT = """You are a content strategist. Transform the research summary 
below into a well-structured content brief.

Topic: {query}

Research Summary:
{research_summary}

Create a structured content brief with:
1. Executive Summary (2-3 sentences)
2. Key Findings (5 bullet points)
3. Target Audience
4. Key Messages (3 main points)
5. Recommended Keywords (5-8 SEO keywords)
6. Content Angles (3 different ways to approach this topic)

Format it cleanly and professionally."""


def content_strategist_agent(state: ContentState) -> dict:
    """
    Takes research summary and creates a structured content brief.
    """
    research_summary = state.get("research_summary")

    if not research_summary:
        return {
            "error": "No research summary found. Run research agent first.",
            "agent_path": [*state["agent_path"], "content_strategist"]
        }

    response = llm.invoke(
        STRATEGIST_PROMPT.format(
            query=state["user_query"],
            research_summary=research_summary
        )
    )

    # Extract keywords from the response
    content = response.content
    keywords = extract_keywords(content)

    return {
        "research_summary": content,
        "seo_keywords": keywords,
        "agent_path": [*state["agent_path"], "content_strategist"]
    }


def extract_keywords(content: str) -> list[str]:
    """
    Extracts keywords from the content brief.
    Looks for the 'Recommended Keywords' section.
    """
    keywords = []
    lines = content.split("\n")
    in_keywords_section = False

    for line in lines:
        if "recommended keywords" in line.lower():
            in_keywords_section = True
            continue
        if in_keywords_section:
            # Stop at next numbered section
            if line.strip() and line.strip()[0].isdigit():
                break
            # Extract keyword from bullet point
            if line.strip().startswith("-") or line.strip().startswith("•"):
                keyword = line.strip().lstrip("-•").strip()
                if keyword:
                    keywords.append(keyword)

    return keywords