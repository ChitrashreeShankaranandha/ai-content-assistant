import json
from langchain_openai import ChatOpenAI
from src.core.config import config
from src.workflow.state_management import ContentState

llm = ChatOpenAI(
    model=config.FAST_MODEL,
    api_key=config.OPENAI_API_KEY,
    temperature=0
)

ROUTER_PROMPT = """You are a content marketing assistant router.
Analyze the user's CURRENT query along with the CONVERSATION HISTORY and determine:
1. What content type they want now
2. Whether this is a follow-up that builds on previous content

CONVERSATION HISTORY:
{history}

CURRENT QUERY: {query}

Return ONLY a JSON object (no markdown, no extra text) with this exact format:
{{"intent": "<intent>", "is_followup": <true_or_false>}}

Valid intents:
- research      → ONLY information/research on a topic, no content to be written
- blog          → ONLY a blog post requested (no LinkedIn, no image)
- linkedin      → ONLY a LinkedIn post requested (no blog, no image)
- image         → ONLY an image requested
- blog_linkedin → BOTH a blog post AND a LinkedIn post, but NO image
- full_content  → ALL of: blog + LinkedIn + image (user explicitly mentions image/visual/picture or says "everything")

is_followup rules:
- true  → user references previous content (e.g. "now write a blog from that", "make it shorter", "add hashtags", "based on the research above")
- false → user asks about a brand new topic

Examples:
- History: "Research AI trends" / Current: "Now write a blog from that" → {{"intent": "blog", "is_followup": true}}
- History: "Research AI trends" / Current: "Now research quantum computing" → {{"intent": "research", "is_followup": false}}
- History: (empty) / Current: "Write a LinkedIn post about climate tech" → {{"intent": "linkedin", "is_followup": false}}
- History: "Write a blog about remote work" / Current: "Now make a LinkedIn version" → {{"intent": "linkedin", "is_followup": true}}

JSON:"""


def format_history(messages: list) -> str:
    """Format previous messages for the router prompt."""
    if not messages or len(messages) <= 1:
        return "(no previous conversation)"
    # Exclude the current user message (the last one)
    prior = messages[:-1] if messages[-1].get("role") == "user" else messages
    formatted = []
    for msg in prior[-6:]:  # only last 6 messages to keep prompt short
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Truncate long messages
        if len(content) > 200:
            content = content[:200] + "..."
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


def query_handler_agent(state: ContentState) -> dict:
    """
    Reads the user query + conversation history.
    Determines intent AND whether this is a follow-up to previous turns.
    """
    query = state["user_query"]
    history = format_history(state.get("messages", []))

    response = llm.invoke(ROUTER_PROMPT.format(history=history, query=query))
    content = response.content.strip()

    # Try to parse JSON response
    intent = "research"
    is_followup = False
    try:
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        parsed = json.loads(content)
        intent = parsed.get("intent", "research").lower()
        is_followup = bool(parsed.get("is_followup", False))
    except (json.JSONDecodeError, ValueError):
        intent = "research"
        is_followup = False

    valid_intents = ["research", "blog", "linkedin", "image", "blog_linkedin", "full_content"]
    if intent not in valid_intents:
        intent = "research"

    return {
        "intent": intent,
        "is_followup": is_followup,
        "agent_path": [*state["agent_path"], "query_handler"]
    }