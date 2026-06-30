# Let's import "os" module, which stands for "Operating System"
# The os module in Python provides a way to interact with the operating system for things like:
# (1) accessing Environment Variables
# (2) Creating, renaming, and deleting files/folders.
import os
from IPython.display import display, Markdown
 # Import the OpenAI API client
from openai import OpenAI 
# Import the Agent class to create and manage AI agents
from agents import Agent
# Import the Runner class, which is used to run an agent and get its output
from agents import Runner
# This will be used to load the API key from the .env file
from dotenv import load_dotenv

load_dotenv()

# Get the OpenAI API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Let's configure the OpenAI Client using our key
openai_client = OpenAI(api_key = openai_api_key)

# Define the instructions for the fact-checker AI Agent

executive_assistant_instructions = """
Context:

You are an elite Executive Assistant Agent.

Instructions:
Your job is to analyze chaotic project updates, meeting notes, or emails, filter out the noise, 
and extract concrete action items. Be aggressive about finding hidden tasks that people agreed to do.

Input:
You will receive meeting notes that you need to parse and extract concrete action items.

Output:
Respond with:
Action items in a format - Action Items - Responsible Person - Deadline - Priority (High/Medium/Low)
"""

# Create a new agent called "Fact Checker"
executive_assistant_agent = Agent(name = "Executive Assistant",   # Name of the agent
                           instructions = executive_assistant_instructions, # The rules and behavior for the agent
                           model = "gpt-4.1-mini") # The AI model (LLM) to use

# Print a confirmation message that the agent was created
print(f"Agent '{executive_assistant_agent.name}' created successfully!")

# A meeting note from which we need to extract action items.
meeting_notes = """
Hey team, quick update from the morning sync. Dave, we absolutely need the Q3 budget slides 
ready by Friday morning because Sarah is reviewing them with investors. Speaking of Sarah, she 
mentioned she likes the new logo direction, but someone needs to send her the high-res PNGs by 
tonight. I'll handle booking the conference room for next week. Also, don't forget we need to update the 
readme file on GitHub eventually, maybe next month.
"""

import asyncio
 

async def main():
    # Replaced print_markdown with standard print

    # Run the Executive Assistant agent on the input meeting notes
    response = await Runner.run(
        starting_agent = executive_assistant_agent,  # The agent we created earlier
        input = meeting_notes               # The meeting notes we want it to process
    )

    # Display the agent's response
    print("\n🤖 Agent's Response:\n")
    print(response.final_output)    # Shows the extracted action items

# This kicks off the asynchronous event loop in a standard Python execution
if __name__ == "__main__":
    asyncio.run(main())

