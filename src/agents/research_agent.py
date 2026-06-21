import requests
from langchain_openai import ChatOpenAI
from src.core.config import config
from src.workflow.state_management import ContentState

llm = ChatOpenAI(
    model=config.PRIMARY_MODEL,
    api_key=config.OPENAI_API_KEY,
    temperature=0.3
)


def search_web(query: str) -> list[dict]:
    """Calls SERP API to get search results for a query."""
    params = {
        "q": query,
        "api_key": config.SERP_API_KEY,
        "num": config.MAX_RESEARCH_RESULTS,
        "engine": "google"
    }
    response = requests.get("https://serpapi.com/search", params=params)
    data = response.json()

    results = []
    for r in data.get("organic_results", []):
        results.append({
            "title": r.get("title", ""),
            "snippet": r.get("snippet", ""),
            "link": r.get("link", "")
        })
    return results


SUMMARIZE_PROMPT = """You are a research analyst. Based on the search results below,
write a comprehensive research summary about: {query}

Search Results:
{results}

Write a well-structured summary with key findings, trends, and insights.
Include source references where relevant."""


def research_agent(state: ContentState) -> dict:
    """
    Searches the web for the user query and summarizes findings.
    """
    query = state["user_query"]

    # Step 1: Search the web
    results = search_web(query)

    if not results:
        return {
            "error": "No search results found",
            "agent_path": [*state["agent_path"], "research_agent"]
        }

    # Step 2: Summarize with LLM
    results_text = "\n\n".join([
        f"Title: {r['title']}\nSnippet: {r['snippet']}\nSource: {r['link']}"
        for r in results
    ])

    response = llm.invoke(
        SUMMARIZE_PROMPT.format(query=query, results=results_text)
    )

    return {
        "research_results": results,
        "research_summary": response.content,
        "agent_path": [*state["agent_path"], "research_agent"]
    }