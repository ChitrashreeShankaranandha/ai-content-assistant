"""
Image Generator Agent.

Primary: FLUX.1-schnell via Hugging Face Inference (free, fast ~5-10s)
Fallback: OpenAI gpt-image-1 (paid)
"""

import requests
import base64
from pathlib import Path
from langchain_openai import ChatOpenAI
from src.core.config import config
from src.workflow.state_management import ContentState

llm = ChatOpenAI(
    model=config.FAST_MODEL,
    api_key=config.OPENAI_API_KEY,
    temperature=0.8
)

# Hugging Face FLUX.1-schnell endpoint (free, fast generation)
HF_API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

PROMPT_ENHANCER = """You are an expert at writing image generation prompts.
Convert this content topic into a detailed, vivid image prompt.

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


def generate_with_flux(prompt: str) -> bytes:
    """Calls Hugging Face FLUX.1-schnell, returns image bytes."""
    headers = {
        "Authorization": f"Bearer {config.HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"num_inference_steps": 25}
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise Exception(f"HF API error: {response.status_code} - {response.text[:300]}")

    if len(response.content) < 1000:
        raise Exception(f"HF returned suspiciously small response: {response.text[:300]}")

    return response.content


def generate_with_dalle(prompt: str, conversation_id: str) -> str:
    """Fallback: generates image using OpenAI gpt-image-1, saves locally, returns file path."""
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=config.IMAGE_SIZE,
        n=1
    )

    image_b64 = response.data[0].b64_json
    image_bytes = base64.b64decode(image_b64)

    filename = f"image_{conversation_id}.png"
    return save_image(image_bytes, filename)


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
    Primary: FLUX.1-schnell via Hugging Face (free)
    Fallback: OpenAI gpt-image-1 (paid)
    """
    query = state["user_query"]

    enhanced_prompt = enhance_prompt(query)

    image_url = None
    image_local_path = None
    fallback_used = None
    image_error = None

    try:
        image_bytes = generate_with_flux(enhanced_prompt)
        filename = f"image_{state['conversation_id']}.png"
        image_local_path = save_image(image_bytes, filename)
        fallback_used = "flux-schnell"

    except Exception as e:
        try:
            image_local_path = generate_with_dalle(enhanced_prompt, state["conversation_id"])
            fallback_used = "gpt-image-1"
        except Exception as e2:
            image_error = f"All image services failed. FLUX: {e}. OpenAI: {e2}"

    return {
        "image_prompt": enhanced_prompt,
        "image_url": image_url,
        "image_local_path": image_local_path,
        "fallback_used": fallback_used,
        "error": image_error,
        "agent_path": [*state["agent_path"], "image_generator"]
    }