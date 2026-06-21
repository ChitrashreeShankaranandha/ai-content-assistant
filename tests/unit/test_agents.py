import pytest
from unittest.mock import patch, MagicMock
from src.workflow.state_management import create_initial_state


class TestQueryHandler:

    @patch("src.agents.query_handler.llm")
    def test_routes_blog_intent(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="blog")
        from src.agents.query_handler import query_handler_agent
        state = create_initial_state("Write a blog about AI", "test-123")
        result = query_handler_agent(state)
        assert result["intent"] == "blog"

    @patch("src.agents.query_handler.llm")
    def test_routes_research_intent(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="research")
        from src.agents.query_handler import query_handler_agent
        state = create_initial_state("Research quantum computing", "test-123")
        result = query_handler_agent(state)
        assert result["intent"] == "research"

    @patch("src.agents.query_handler.llm")
    def test_invalid_intent_defaults_to_research(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="something_random")
        from src.agents.query_handler import query_handler_agent
        state = create_initial_state("Do something", "test-123")
        result = query_handler_agent(state)
        assert result["intent"] == "research"

    @patch("src.agents.query_handler.llm")
    def test_agent_path_updated(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="blog")
        from src.agents.query_handler import query_handler_agent
        state = create_initial_state("Write a blog", "test-123")
        result = query_handler_agent(state)
        assert "query_handler" in result["agent_path"]