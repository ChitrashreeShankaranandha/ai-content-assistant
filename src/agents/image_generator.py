import requests
import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from src.core.config import config
from src.workflow.state_management import ContentState

llm = ChatOpenAI(
    model=config.FAST_MODEL,
    api_key=config.OPENAI_API_KEY,
    temperature=0.8
)

# Hugging Face SDXL endpoint
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

PROMPT_ENHANCER = """You are an expert at writing image generation prompts.
Convert this content topic into a detailed, vivid image prompt for Stable Diffusion.

Topic: {query}

Requirements:
- Describe a professional marketing image
- Include style (photorealistic, digital art, etc.)
- Include lighting, mood, and composition
- Keep it under 200 words
- No text or words in the image
- Suitable for a professional blog or LinkedIn post

Return ONLY the image prompt, nothing else."""


def enhance_prompt(query: str) -> str:
    """Uses LLM to convert topic into a detailed image generation prompt."""
    response = llm.invoke(PROMPT_ENHANCER.format(query=query))
    return response.content.strip()


def generate_with_sdxl(prompt: str) -> bytes:
    """Calls Hugging Face SDXL API and returns image bytes."""
    headers = {"Authorization": f"Bearer {config.HF_API_TOKEN}"}
    payload = {"inputs": prompt}

    response = requests.post(HF_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"HF API error: {response.status_code} - {response.text}")

    return response.content


def generate_with_dalle(prompt: str) -> str:
    """Fallback: generates image using DALL-E 3, returns URL."""
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=config.IMAGE_SIZE,
        quality="standard",
        n=1
    )
    return response.data[0].url


def save_image(image_bytes: bytes, filename: str) -> str:
    """Saves image bytes to local file, returns file path."""
    output_dir = Path(config.IMAGE_OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / filename
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    return str(filepath)


def image_generator_agent(state: ContentState) -> dict:
    """
    Generates an image for the content topic.
    Primary: Stable Diffusion XL via Hugging Face
    Fallback: DALL-E 3 via OpenAI
    """
    query = state["user_query"]

    # Step 1: Enhance the prompt
    enhanced_prompt = enhance_prompt(query)

    # Step 2: Try SDXL first, fall back to DALL-E 3
    image_url = None
    image_local_path = None
    fallback_used = None

    try:
        image_bytes = generate_with_sdxl(enhanced_prompt)
        filename = f"image_{state['conversation_id']}.png"
        image_local_path = save_image(image_bytes, filename)

    except Exception as e:
        print(f"SDXL failed: {e}. Falling back to DALL-E 3...")
        fallback_used = "dall-e-3"
        image_url = generate_with_dalle(enhanced_prompt)

    return {
        "image_prompt": enhanced_prompt,
        "image_url": image_url,
        "image_local_path": image_local_path,
        "fallback_used": fallback_used,
        "agent_path": [*state["agent_path"], "image_generator"]
    }