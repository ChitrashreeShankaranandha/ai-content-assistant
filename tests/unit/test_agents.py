import pytest
from unittest.mock import patch, MagicMock
from src.workflow.state_management import create_initial_state


class TestQueryHandler:

    @patch("src.agents.query_handler.llm")
    def test_routes_blog_intent(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content='{"intent": "blog", "is_followup": false}')
        from src.agents.query_handler import query_handler_agent
        state = create_initial_state("Write a blog about AI", "test-123")
        result = query_handler_agent(state)
        assert result["intent"] == "blog"

    @patch("src.agents.query_handler.llm")
    def test_routes_research_intent(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content='{"intent": "research", "is_followup": false}')
        from src.agents.query_handler import query_handler_agent
        state = create_initial_state("Research quantum computing", "test-123")
        result = query_handler_agent(state)
        assert result["intent"] == "research"

    @patch("src.agents.query_handler.llm")
    def test_invalid_intent_defaults_to_research(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content='{"intent": "something_random", "is_followup": false}')
        from src.agents.query_handler import query_handler_agent
        state = create_initial_state("Do something", "test-123")
        result = query_handler_agent(state)
        assert result["intent"] == "research"

    @patch("src.agents.query_handler.llm")
    def test_agent_path_updated(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content='{"intent": "blog", "is_followup": false}')
        from src.agents.query_handler import query_handler_agent
        state = create_initial_state("Write a blog", "test-123")
        result = query_handler_agent(state)
        assert "query_handler" in result["agent_path"]
    
    @patch("src.agents.query_handler.llm")
    def test_detects_followup(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content='{"intent": "linkedin", "is_followup": true}')
        from src.agents.query_handler import query_handler_agent
        state = create_initial_state("Now write a LinkedIn post from that", "test-123")
        state["messages"] = [
            {"role": "user", "content": "Research AI trends"},
            {"role": "assistant", "content": "Here is research on AI trends..."},
            {"role": "user", "content": "Now write a LinkedIn post from that"}
        ]
        result = query_handler_agent(state)
        assert result["is_followup"] is True
        assert result["intent"] == "linkedin"


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


class TestContentStrategist:

    @patch("src.agents.content_strategist.llm")
    def test_returns_structured_content(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="1. Executive Summary\nGreat topic.\n"
                    "2. Key Findings\n- Finding 1\n"
                    "5. Recommended Keywords\n- AI trends\n- machine learning\n"
                    "6. Content Angles\n- Angle 1"
        )
        from src.agents.content_strategist import content_strategist_agent
        state = create_initial_state("AI trends", "test-123")
        state["research_summary"] = "AI is growing fast"
        result = content_strategist_agent(state)
        assert result["content_brief"] is not None

    @patch("src.agents.content_strategist.llm")
    def test_extracts_keywords(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="5. Recommended Keywords\n- AI trends\n- machine learning\n6. Content Angles"
        )
        from src.agents.content_strategist import content_strategist_agent
        state = create_initial_state("AI trends", "test-123")
        state["research_summary"] = "AI is growing fast"
        result = content_strategist_agent(state)
        assert isinstance(result["seo_keywords"], list)

    def test_handles_missing_research_summary(self):
        from src.agents.content_strategist import content_strategist_agent
        state = create_initial_state("AI trends", "test-123")
        result = content_strategist_agent(state)
        assert result["error"] is not None

    @patch("src.agents.content_strategist.llm")
    def test_agent_path_updated(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="Structured content brief")
        from src.agents.content_strategist import content_strategist_agent
        state = create_initial_state("AI trends", "test-123")
        state["research_summary"] = "AI is growing fast"
        result = content_strategist_agent(state)
        assert "content_strategist" in result["agent_path"]


class TestBlogWriter:

    @patch("src.agents.blog_writer.llm")
    def test_writes_blog_post(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="# AI Trends 2025\n\nAI is transforming industries..."
        )
        from src.agents.blog_writer import blog_writer_agent
        state = create_initial_state("AI trends 2025", "test-123")
        state["research_summary"] = "AI is growing fast"
        state["seo_keywords"] = ["AI trends", "machine learning"]
        result = blog_writer_agent(state)
        assert result["blog_post"] is not None

    @patch("src.agents.blog_writer.llm")
    def test_blog_post_contains_content(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="# AI Trends 2025\n\nAI is transforming industries..."
        )
        from src.agents.blog_writer import blog_writer_agent
        state = create_initial_state("AI trends 2025", "test-123")
        state["research_summary"] = "AI is growing fast"
        state["seo_keywords"] = ["AI trends"]
        result = blog_writer_agent(state)
        assert "AI Trends 2025" in result["blog_post"]

    def test_handles_missing_research_summary(self):
        from src.agents.blog_writer import blog_writer_agent
        state = create_initial_state("AI trends 2025", "test-123")
        result = blog_writer_agent(state)
        assert result["error"] is not None

    @patch("src.agents.blog_writer.llm")
    def test_agent_path_updated(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="# Blog Post")
        from src.agents.blog_writer import blog_writer_agent
        state = create_initial_state("AI trends", "test-123")
        state["research_summary"] = "AI is growing fast"
        result = blog_writer_agent(state)
        assert "blog_writer" in result["agent_path"]

    @patch("src.agents.blog_writer.llm")
    def test_works_without_keywords(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="# Blog Post\n\nContent here")
        from src.agents.blog_writer import blog_writer_agent
        state = create_initial_state("AI trends", "test-123")
        state["research_summary"] = "AI is growing fast"
        result = blog_writer_agent(state)
        assert result["blog_post"] is not None


class TestLinkedInWriter:

    @patch("src.agents.linkedin_writer.llm")
    def test_writes_linkedin_post(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="AI is changing everything.\n\nHere are 3 things you need to know...\n\n#AI #MachineLearning"
        )
        from src.agents.linkedin_writer import linkedin_writer_agent
        state = create_initial_state("AI trends 2025", "test-123")
        state["research_summary"] = "AI is growing fast"
        state["seo_keywords"] = ["AI trends", "machine learning"]
        result = linkedin_writer_agent(state)
        assert result["linkedin_post"] is not None

    @patch("src.agents.linkedin_writer.llm")
    def test_linkedin_post_contains_hashtags(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="Great insights on AI.\n\n#AI #MachineLearning #Technology"
        )
        from src.agents.linkedin_writer import linkedin_writer_agent
        state = create_initial_state("AI trends 2025", "test-123")
        state["research_summary"] = "AI is growing fast"
        state["seo_keywords"] = ["AI trends"]
        result = linkedin_writer_agent(state)
        assert "#" in result["linkedin_post"]

    def test_handles_missing_research_summary(self):
        from src.agents.linkedin_writer import linkedin_writer_agent
        state = create_initial_state("AI trends 2025", "test-123")
        result = linkedin_writer_agent(state)
        assert result["error"] is not None

    @patch("src.agents.linkedin_writer.llm")
    def test_agent_path_updated(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="LinkedIn post content #AI")
        from src.agents.linkedin_writer import linkedin_writer_agent
        state = create_initial_state("AI trends", "test-123")
        state["research_summary"] = "AI is growing fast"
        result = linkedin_writer_agent(state)
        assert "linkedin_writer" in result["agent_path"]

    @patch("src.agents.linkedin_writer.llm")
    def test_works_without_keywords(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="AI is transforming business.\n\n#AI #Innovation"
        )
        from src.agents.linkedin_writer import linkedin_writer_agent
        state = create_initial_state("AI trends", "test-123")
        state["research_summary"] = "AI is growing fast"
        result = linkedin_writer_agent(state)
        assert result["linkedin_post"] is not None


class TestImageGenerator:

    @patch("src.agents.image_generator.generate_with_flux")
    @patch("src.agents.image_generator.enhance_prompt")
    @patch("src.agents.image_generator.save_image")
    def test_generates_image_with_flux(
        self, mock_save, mock_enhance, mock_flux
    ):
        mock_enhance.return_value = "A professional image of AI technology"
        mock_flux.return_value = b"fake_image_bytes"
        mock_save.return_value = "generated_images/image_test-123.png"

        from src.agents.image_generator import image_generator_agent
        state = create_initial_state("AI trends", "test-123")
        result = image_generator_agent(state)

        assert result["image_prompt"] is not None
        assert result["image_local_path"] is not None
        assert result["fallback_used"] == "flux-schnell"

    @patch("src.agents.image_generator.generate_with_dalle")
    @patch("src.agents.image_generator.generate_with_flux")
    @patch("src.agents.image_generator.enhance_prompt")
    def test_falls_back_to_openai_when_flux_fails(
        self, mock_enhance, mock_flux, mock_dalle
    ):
        mock_enhance.return_value = "A professional image of AI"
        mock_flux.side_effect = Exception("HF API error")
        mock_dalle.return_value = "generated_images/image_test-123.png"

        from src.agents.image_generator import image_generator_agent
        state = create_initial_state("AI trends", "test-123")
        result = image_generator_agent(state)

        assert result["fallback_used"] == "gpt-image-1"
        assert result["image_local_path"] == "generated_images/image_test-123.png"

    @patch("src.agents.image_generator.generate_with_flux")
    @patch("src.agents.image_generator.enhance_prompt")
    @patch("src.agents.image_generator.save_image")
    def test_agent_path_updated(
        self, mock_save, mock_enhance, mock_flux
    ):
        mock_enhance.return_value = "Professional AI image"
        mock_flux.return_value = b"fake_image_bytes"
        mock_save.return_value = "generated_images/test.png"

        from src.agents.image_generator import image_generator_agent
        state = create_initial_state("AI trends", "test-123")
        result = image_generator_agent(state)

        assert "image_generator" in result["agent_path"]

    @patch("src.agents.image_generator.generate_with_flux")
    @patch("src.agents.image_generator.enhance_prompt")
    @patch("src.agents.image_generator.save_image")
    def test_prompt_is_enhanced(
        self, mock_save, mock_enhance, mock_flux
    ):
        mock_enhance.return_value = "Detailed enhanced prompt for AI image"
        mock_flux.return_value = b"fake_image_bytes"
        mock_save.return_value = "generated_images/test.png"

        from src.agents.image_generator import image_generator_agent
        state = create_initial_state("AI trends", "test-123")
        result = image_generator_agent(state)

        assert result["image_prompt"] == "Detailed enhanced prompt for AI image"