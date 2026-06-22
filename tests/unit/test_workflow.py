import pytest
from unittest.mock import patch, MagicMock
from src.workflow.state_management import create_initial_state


def test_initial_state_has_user_query():
    state = create_initial_state("Write a blog about AI", "test-123")
    assert state["user_query"] == "Write a blog about AI"

def test_initial_state_has_message():
    state = create_initial_state("Write a blog about AI", "test-123")
    assert state["messages"][0]["role"] == "user"

def test_initial_state_intent_is_none():
    state = create_initial_state("Write a blog about AI", "test-123")
    assert state["intent"] is None

def test_initial_state_agent_path_is_empty():
    state = create_initial_state("Write a blog about AI", "test-123")
    assert state["agent_path"] == []

def test_initial_state_conversation_id():
    state = create_initial_state("Write a blog about AI", "test-123")
    assert state["conversation_id"] == "test-123"


class TestWorkflowGraph:

    def test_workflow_builds_successfully(self):
        from src.workflow.langgraph_workflow import workflow
        assert workflow is not None

    def test_route_by_intent_research(self):
        from src.workflow.langgraph_workflow import route_by_intent
        state = create_initial_state("test", "test-123")
        state["intent"] = "research"
        assert route_by_intent(state) == "research"

    def test_route_by_intent_image(self):
        from src.workflow.langgraph_workflow import route_by_intent
        state = create_initial_state("test", "test-123")
        state["intent"] = "image"
        assert route_by_intent(state) == "image_generator"

    def test_route_by_intent_blog_goes_to_research_first(self):
        from src.workflow.langgraph_workflow import route_by_intent
        state = create_initial_state("test", "test-123")
        state["intent"] = "blog"
        assert route_by_intent(state) == "research"

    def test_route_after_research_for_research_only(self):
        from src.workflow.langgraph_workflow import route_after_research
        from langgraph.graph import END
        state = create_initial_state("test", "test-123")
        state["intent"] = "research"
        assert route_after_research(state) == END

    def test_route_after_strategist_for_blog(self):
        from src.workflow.langgraph_workflow import route_after_strategist
        state = create_initial_state("test", "test-123")
        state["intent"] = "blog"
        assert route_after_strategist(state) == "blog_writer"

    def test_route_after_strategist_for_linkedin(self):
        from src.workflow.langgraph_workflow import route_after_strategist
        state = create_initial_state("test", "test-123")
        state["intent"] = "linkedin"
        assert route_after_strategist(state) == "linkedin_writer"

    def test_route_after_blog_for_full_content(self):
        from src.workflow.langgraph_workflow import route_after_blog
        state = create_initial_state("test", "test-123")
        state["intent"] = "full_content"
        assert route_after_blog(state) == "linkedin_writer"