from langchain_openai import ChatOpenAI
from src.core.config import config
from src.workflow.state_management import ContentState

llm = ChatOpenAI(
    model=config.FAST_MODEL,
    api_key=config.OPENAI_API_KEY,
    temperature=0
)

ROUTER_PROMPT = """You are a content marketing assistant router.
Analyze the user's query and determine which agent should handle it.

Return ONLY one of these intents (no explanation):
- research     → user wants information/research on a topic
- blog         → user wants a blog post written
- linkedin     → user wants a LinkedIn post written
- image        → user wants an image generated
- full_content → user wants research + blog + linkedin + image all together

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
    valid_intents = ["research", "blog", "linkedin", "image", "full_content"]
    if intent not in valid_intents:
        intent = "research"  # safe default

    return {
        "intent": intent,
        "agent_path": [*state["agent_path"], "query_handler"]
    }