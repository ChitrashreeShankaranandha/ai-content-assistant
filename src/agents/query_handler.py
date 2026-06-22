from langchain_openai import ChatOpenAI
from src.core.config import config
from src.workflow.state_management import ContentState

llm = ChatOpenAI(
    model=config.FAST_MODEL,
    api_key=config.OPENAI_API_KEY,
    temperature=0
)

ROUTER_PROMPT = """You are a content marketing assistant router.
Analyze the user's query and determine which agent(s) should handle it.

Return ONLY ONE of these intents (no explanation, no extra text):

- research      → ONLY information/research on a topic, no content to be written
- blog          → ONLY a blog post requested (no LinkedIn, no image)
- linkedin      → ONLY a LinkedIn post requested (no blog, no image)
- image         → ONLY an image requested (no text content)
- blog_linkedin → BOTH a blog post AND a LinkedIn post, but NO image
- full_content  → ALL of: blog + LinkedIn + image (user must explicitly mention image, visual, picture, or say "everything"/"all")

Important rules:
- Only return "full_content" if the user EXPLICITLY asks for an image/visual/picture, or says "everything" or "all content"
- If the user asks for blog AND linkedin but does NOT mention image, return "blog_linkedin"
- If the user asks for only one content type, return that specific intent

User query: {query}

Intent:"""


def query_handler_agent(state: ContentState) -> dict:
    """
    Reads the user query and determines intent.
    Returns intent and updates agent_path.
    """
    query = state["user_query"]

    response = llm.invoke(ROUTER_PROMPT.format(query=query))
    intent = response.content.strip().lower()

    # Validate intent
    valid_intents = ["research", "blog", "linkedin", "image", "blog_linkedin", "full_content"]
    if intent not in valid_intents:
        intent = "research"  # safe default

    return {
        "intent": intent,
        "agent_path": [*state["agent_path"], "query_handler"]
    }