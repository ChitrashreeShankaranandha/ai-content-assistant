import pytest
from src.core.config import config

def test_openai_key_loaded():
    assert config.OPENAI_API_KEY is not None

def test_default_model_is_gpt4o():
    assert config.PRIMARY_MODEL == "gpt-4o"

def test_default_environment_is_development():
    assert config.APP_ENV == "development"

def test_cache_ttl_default():
    assert config.CACHE_TTL == 1800