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
from src.workflow.langgraph_workflow import workflow
from src.workflow.state_management import create_initial_state


# ---------- Page setup ----------
st.set_page_config(
    page_title="ContentAlchemy",
    page_icon="✨",
    layout="wide"
)

st.title("✨ ContentAlchemy")
st.caption("Your AI-powered multi-agent content marketing assistant")


# ---------- Session state initialization ----------
# Streamlit reruns the script on every interaction.
# st.session_state persists data across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())


# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **ContentAlchemy** is a multi-agent AI system that creates content for you.

    **Try asking:**
    - *Research the latest AI trends*
    - *Write a blog post about remote work*
    - *Create a LinkedIn post about agentic AI*
    - *Generate an image of a futuristic city*
    - *Write a blog and LinkedIn post about climate tech*
    - *Create everything about quantum computing* (full content)
    """)

    st.divider()

    if st.button("🔄 New Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
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
            state = create_initial_state(user_query, st.session_state.conversation_id)
            result = workflow.invoke(state)

        # Show agent path
        path = " → ".join(result.get("agent_path", []))
        st.caption(f"🛤️ {path}")

        # Build the response based on what was generated
        response_parts = []

        if result.get("research_summary"):
            response_parts.append("### 📚 Research Summary\n" + result["research_summary"])

        if result.get("blog_post"):
            response_parts.append("### ✍️ Blog Post\n" + result["blog_post"])

        if result.get("linkedin_post"):
            response_parts.append("### 📱 LinkedIn Post\n" + result["linkedin_post"])

        if result.get("error"):
            response_parts.append(f"⚠️ **Error:** {result['error']}")

        if not response_parts:
            response_parts.append("Hmm, I couldn't generate any content. Try rephrasing your request.")

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