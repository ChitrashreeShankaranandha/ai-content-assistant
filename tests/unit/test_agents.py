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
        

class TestResearchAgent:

    @patch("src.agents.research_agent.search_web")
    @patch("src.agents.research_agent.llm")
    def test_research_returns_summary(self, mock_llm, mock_search):
        mock_search.return_value = [
            {"title": "AI Trends", "snippet": "AI is growing fast", "link": "http://example.com"}
        ]
        mock_llm.invoke.return_value = MagicMock(content="AI is transforming industries...")
        from src.agents.research_agent import research_agent
        state = create_initial_state("AI trends 2025", "test-123")
        result = research_agent(state)
        assert result["research_summary"] == "AI is transforming industries..."

    @patch("src.agents.research_agent.search_web")
    @patch("src.agents.research_agent.llm")
    def test_research_stores_results(self, mock_llm, mock_search):
        mock_search.return_value = [
            {"title": "AI Trends", "snippet": "AI is growing", "link": "http://example.com"}
        ]
        mock_llm.invoke.return_value = MagicMock(content="Summary here")
        from src.agents.research_agent import research_agent
        state = create_initial_state("AI trends 2025", "test-123")
        result = research_agent(state)
        assert len(result["research_results"]) == 1

    @patch("src.agents.research_agent.search_web")
    def test_research_handles_no_results(self, mock_search):
        mock_search.return_value = []
        from src.agents.research_agent import research_agent
        state = create_initial_state("AI trends 2025", "test-123")
        result = research_agent(state)
        assert result["error"] == "No search results found"

    @patch("src.agents.research_agent.search_web")
    @patch("src.agents.research_agent.llm")
    def test_agent_path_updated(self, mock_llm, mock_search):
        mock_search.return_value = [
            {"title": "Test", "snippet": "Test snippet", "link": "http://test.com"}
        ]
        mock_llm.invoke.return_value = MagicMock(content="Summary")
        from src.agents.research_agent import research_agent
        state = create_initial_state("AI trends", "test-123")
        result = research_agent(state)
        assert "research_agent" in result["agent_path"]