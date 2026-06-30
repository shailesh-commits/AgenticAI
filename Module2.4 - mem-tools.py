import os
import asyncio
from typing import Literal
from pydantic import BaseModel, Field
from openai import OpenAI 
from dotenv import load_dotenv

# Maintaining consistency with your internal framework imports
from agents import Agent, SQLiteSession, Runner

load_dotenv()

# Get the OpenAI API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

# ==========================================
# 1. MOCK DATA BACKEND & CORE FUNCTION
# ==========================================

MOCK_LICENSE_DATABASE = {
    "acme corp": {"chatgpt": 450, "claude": 120, "github_copilot": 300},
    "dev studio": {"chatgpt": 25, "claude": 85, "github_copilot": 10},
    "global tech": {"chatgpt": 1200, "claude": 1100, "github_copilot": 1500}
}

# Define the argument validation schema using Pydantic
class LicenseQueryArgs(BaseModel):
    organization_name: str = Field(
        description="The full canonical name of the organization or company to look up (e.g., 'Acme Corp')."
    )
    provider_type: Literal["chatgpt", "claude", "github_copilot", "all"] = Field(
        default="all",
        description="The specific AI platform provider to look up. Use 'all' to retrieve the complete breakdown."
    )

# The actual Python execution logic for the tool
def fetch_organization_licenses(organization_name: str, provider_type: str = "all") -> str:
    """Queries the internal software asset management database to retrieve 
    allocated active user license counts for a target organization."""
    
    org_key = organization_name.lower().strip()
    org_data = MOCK_LICENSE_DATABASE.get(org_key)
    
    if not org_data:
        return f"Error: Organization '{organization_name}' not found in asset management system."
        
    if provider_type == "all":
        return f"Active license allocation for {organization_name}: {org_data}"
        
    count = org_data.get(provider_type, 0)
    return f"Active seat allocation for {organization_name} ({provider_type}): {count} licenses."


# ==========================================
# 2. UPDATED AGENT SYSTEM INSTRUCTIONS
# ==========================================

knowledgebot_instructions = """
Context:
You are an expert AI enterprise operations tutor and infrastructure analyst.

Instructions:
1. Provide concise, accurate factual answers based on your knowledge base.
2. If the user asks about software licenses, corporate allocations, seat usage, or provider distributions for an organization, you MUST use the `fetch_organization_licenses` tool.
3. Do not guess license metrics or allocate imaginary values. Rely strictly on the tool's output observation.
4. If you don't know the answer and lack a tool to discover it, explicitly state "I don't know".
"""

# Create the agent, passing the tool and its validation schema explicitly
knowledgebot_agent = Agent(
    name="Knowledge Bot",
    instructions=knowledgebot_instructions,
    model="gpt-4.1-mini",
    # Registering the tool wrapper array into your SDK specification
    tools=[(fetch_organization_licenses, LicenseQueryArgs)] 
)

print(f"Agent '{knowledgebot_agent.name}' initialized with Tool integrations successfully!")

# Instantiate state persistence session storage
session = SQLiteSession("conversation_with_tools")

async def main(questiontext: str) -> str:
    response = await Runner.run(    
        starting_agent=knowledgebot_agent,
        input=questiontext,
        session=session
    )
    return response.final_output

# ==========================================
# 3. LIVE WORKFLOW TRACE SESSIONS
# ==========================================

if __name__ == "__main__":
    # Question 1: Triggers the tool call logic path dynamically
    question1 = "How many ChatGPT seats is Acme Corp currently paying for?"
    print("\n🤖 User Prompt:", question1)
    print("🤖 Agent's Response:")
    print(asyncio.run(main(question1)))
    
    # Question 2: Relies on short-term session state to perform a contextual follow-up
    nextquestion = "How does that compare to their Claude license deployment?"
    print("\n🤖 User Prompt:", nextquestion)
    print("🤖 Agent's Response:")
    print(asyncio.run(main(nextquestion)))