"""
Quick demo script to test the full workflow with real APIs.
Run with: python run_demo.py
"""
import uuid
from src.workflow.langgraph_workflow import workflow
from src.workflow.state_management import create_initial_state


def run_query(query: str):
    print("=" * 70)
    print(f"USER QUERY: {query}")
    print("=" * 70)

    state = create_initial_state(query, str(uuid.uuid4()))
    result = workflow.invoke(state)

    print(f"\n📍 Intent: {result.get('intent')}")
    print(f"🛤️  Agent path: {' → '.join(result.get('agent_path', []))}")

    if result.get("research_summary"):
        print(f"\n📚 RESEARCH SUMMARY:\n{result['research_summary'][:500]}...")

    if result.get("blog_post"):
        print(f"\n✍️ BLOG POST (first 500 chars):\n{result['blog_post'][:500]}...")

    if result.get("linkedin_post"):
        print(f"\n📱 LINKEDIN POST:\n{result['linkedin_post']}")

        print(f"\n🎨 IMAGE INFO:")
        print(f"   Prompt: {result.get('image_prompt')}")
        print(f"   Local path: {result.get('image_local_path')}")
        print(f"   URL: {result.get('image_url')}")
        print(f"   Fallback used: {result.get('fallback_used')}")
        if result.get("error"):
            print(f"   Image error: {result.get('error')}")

    if result.get("error"):
        print(f"\n❌ ERROR: {result['error']}")

    print("\n" + "=" * 70 + "\n")


# if __name__ == "__main__":
#     # Try one simple query first
#     run_query("Research the latest trends in AI agents in 2025")


if __name__ == "__main__":
    # Full pipeline: research → strategist → blog → linkedin → image
    run_query("Generate an image of Veg Biriyani")