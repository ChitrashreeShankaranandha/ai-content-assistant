import pytest
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