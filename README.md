---
title: AI Content Assistant
emoji: ✨
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.38.0
app_file: src/web_app/streamlit_app.py
pinned: false
license: mit
---

# ✨ AI Content Assistant

A multi-agent AI content marketing assistant built with LangGraph, OpenAI, and Streamlit.

## Features

- 🧭 **Query Handler** — Smart intent routing
- 🔍 **Research Agent** — Web search via SERP API
- 📋 **Content Strategist** — Structures research into briefs
- ✍️ **Blog Writer** — SEO-optimized long-form content
- 📱 **LinkedIn Writer** — Engagement-optimized social posts
- 🎨 **Image Generator** — FLUX.1-schnell (primary), gpt-image-1 (fallback)

## Tech Stack

- **Multi-Agent Orchestration**: LangGraph
- **LLM**: OpenAI GPT-4o
- **Research**: SERP API
- **Image Generation**: Hugging Face FLUX.1-schnell + OpenAI gpt-image-1
- **Frontend**: Streamlit

## Try It

Examples:
- *"Research the latest AI trends"*
- *"Write a blog about remote work"*
- *"Create everything about quantum computing"*