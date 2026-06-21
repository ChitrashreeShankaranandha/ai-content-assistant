import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PRIMARY_MODEL = "gpt-4o"
    FAST_MODEL = "gpt-4o-mini"
    MAX_TOKENS = 4000
    TEMPERATURE = 0.7

    # Research
    SERP_API_KEY = os.getenv("SERP_API_KEY")
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    MAX_RESEARCH_RESULTS = int(os.getenv("MAX_RESEARCH_RESULTS", "5"))

    # Image Generation
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    IMAGE_SIZE = "1024x1024"
    IMAGE_OUTPUT_DIR = "generated_images"

    # App
    APP_ENV = os.getenv("APP_ENV", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    CACHE_TTL = int(os.getenv("CONTENT_CACHE_TTL", "1800"))

config = Config()