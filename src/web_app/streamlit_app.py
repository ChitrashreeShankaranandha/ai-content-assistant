"""
ContentAlchemy — Streamlit Chat Interface
Run with: streamlit run src/web_app/streamlit_app.py
"""

import sys
import uuid
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from src.workflow.langgraph_workflow import get_workflow
from langgraph.checkpoint.memory import MemorySaver
from src.workflow.state_management import create_initial_state


# ---------- Page setup ----------
st.set_page_config(
    page_title="ContentAlchemy",
    page_icon="✨",
    layout="wide"
)

# Centered title block with custom styling
st.markdown(
    """
    <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
        <h1 style="color: #1f77b4; font-size: 3rem; margin-bottom: 0.2rem;">
            ✨ AI ContentAlchemy
        </h1>
        <p style="font-size: 1.1rem; color: #555; margin-bottom: 0.2rem;">
            Your AI-powered content marketing studio
        </p>
        <p style="font-size: 1.1rem; color: #555; margin-bottom: 0.2rem;">
            Tell it what content you need — it researches, writes, and designs in one go.
        </p>
        <p style="font-size: 0.85rem; color: #888; margin-top: 0.5rem;">
            © 2026 Chitrashree Shankaranandha. All rights reserved.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------- Session state initialization ----------
# Streamlit reruns the script on every interaction.
# st.session_state persists data across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()

if "workflow" not in st.session_state:
    st.session_state.workflow = get_workflow(st.session_state.checkpointer)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("✨ AI Content Assistant")
    st.markdown("""
    ### 🎯 What I can create for you:
    - 🔍 **Research summaries** with sources
    - ✍️ **SEO blog posts** (1200-1500 words)
    - 📱 **LinkedIn posts** with hashtag strategy
    - 🎨 **Marketing images**
    - 📦 **Full content packages** (research + blog + LinkedIn + image)

    ### 💡 Example prompts:
    - *Research the latest AI trends*
    - *Write a blog post about remote work*
    - *Create a LinkedIn post about agentic AI*
    - *Generate an image of a futuristic city*
    - *Write a blog and LinkedIn post about climate tech*
    - *Create everything about quantum computing*

    ### 🔄 Multi-turn chats
    Ask a follow-up like *"now make a LinkedIn post from that research"* — I'll remember the context.
    """)

    st.divider()

    if st.button("🔄 New Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.checkpointer = MemorySaver()  # fresh memory
        st.session_state.workflow = get_workflow(st.session_state.checkpointer)
        st.rerun()


# ---------- Display chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Show text content
        if msg.get("content"):
            st.markdown(msg["content"])
        # Show image if present
        if msg.get("image_path"):
            st.image(msg["image_path"], caption="Generated image")


# ---------- Chat input ----------
if user_query := st.chat_input("Ask me to create content..."):

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Run the workflow
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agents are working..."):
            # When using a checkpointer with thread_id, only pass the NEW data.
            # LangGraph will load saved state for this thread and merge.
            # Passing None values would overwrite saved state.
            history_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            update = {
                "user_query": user_query,
                "messages": history_messages,
                "conversation_id": st.session_state.conversation_id,
                "agent_path": [],  # reset path for this turn
                # Clear per-turn outputs so old ones don't leak through
                "blog_post": None,
                "linkedin_post": None,
                "image_url": None,
                "image_local_path": None,
                "image_prompt": None,
                "error": None,
                "fallback_used": None,
                "intent": None,
                "is_followup": None,
            }

            config = {"configurable": {"thread_id": st.session_state.conversation_id}}
            result = st.session_state.workflow.invoke(update, config=config)

        # Show agent path
        path = " → ".join(result.get("agent_path", []))
        st.caption(f"🛤️ {path}")

        # Debug routing info
        with st.expander("🔍 Debug: Routing"):
            st.write(f"**Intent:** `{result.get('intent')}`")
            st.write(f"**Is follow-up:** `{result.get('is_followup')}`")
            st.write(f"**Has research_summary in state:** `{bool(result.get('research_summary'))}`")
            st.write(f"**Messages in update:** {len(update.get('messages', []))}")

        # Build the response based on what was generated
        response_parts = []

        # Only show research summary if research_agent ran THIS turn
        if "research_agent" in result.get("agent_path", []) and result.get("research_summary"):
            response_parts.append("### 📚 Research Summary\n" + result["research_summary"])

        if result.get("blog_post"):
            response_parts.append("### ✍️ Blog Post\n" + result["blog_post"])

        if result.get("linkedin_post"):
            response_parts.append("### 📱 LinkedIn Post\n" + result["linkedin_post"])

        if result.get("error"):
            response_parts.append(f"⚠️ **Error:** {result['error']}")

        # Check if an image was generated (image-only intent)
        has_image = bool(result.get("image_local_path") or result.get("image_url"))

        if not response_parts and not has_image:
            response_parts.append("Hmm, I couldn't generate any content. Try rephrasing your request.")
        elif not response_parts and has_image:
            response_parts.append("### 🎨 Here's your generated image:")

        full_response = "\n\n---\n\n".join(response_parts)
        st.markdown(full_response)

        # Show image if generated
        image_path = result.get("image_local_path")
        image_url = result.get("image_url")
        fallback = result.get("fallback_used")

        # Debug info
        with st.expander("🔍 Debug: Image generation"):
            st.write(f"**image_local_path:** `{image_path}`")
            st.write(f"**image_url:** `{image_url}`")
            st.write(f"**fallback_used:** `{fallback}`")
            st.write(f"**image_prompt:** {result.get('image_prompt', 'None')[:200] if result.get('image_prompt') else 'None'}...")

        if image_path:
            try:
                st.image(image_path, caption="🎨 Generated image")
            except Exception as e:
                st.error(f"Could not display image: {e}")
        elif image_url:
            st.image(image_url, caption="🎨 Generated image")
        else:
            st.warning("⚠️ No image was generated")

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "image_path": image_path
        })