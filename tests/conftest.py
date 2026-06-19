import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_openai_response():
    """Fake OpenAI response — avoids real API calls in unit tests."""
    mock = MagicMock()
    mock.choices[0].message.content = "This is a mock AI response"
    return mock


@pytest.fixture
def sample_state():
    """Minimal ContentState dict for testing agents in isolation."""
    return {
        "messages": [{"role": "user", "content": "Write a blog about AI"}],
        "user_query": "Write a blog about AI",
        "conversation_id": "test-123",
        "intent": None,
        "agent_path": [],
        "research_results": None,
        "research_summary": None,
        "blog_post": None,
        "linkedin_post": None,
        "image_prompt": None,
        "image_url": None,
        "image_local_path": None,
        "quality_score": None,
        "seo_keywords": None,
        "content_warnings": None,
        "error": None,
        "fallback_used": None,
    }