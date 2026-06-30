# Let's import "os" module, which stands for "Operating System"
# The os module in Python provides a way to interact with the operating system for things like:
# (1) accessing Environment Variables
# (2) Creating, renaming, and deleting files/folders.
import os
import asyncio
from IPython.display import display, Markdown
 # Import the OpenAI API client
from openai import OpenAI 
# Import the Agent class to create and manage AI agents
from agents import Agent, SQLiteSession
# Import the Runner class, which is used to run an agent and get its output
from agents import Runner
# This will be used to load the API key from the .env file
from dotenv import load_dotenv

load_dotenv()

# Get the OpenAI API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Let's configure the OpenAI Client using our key
openai_client = OpenAI(api_key = openai_api_key)

# Define the instructions for the KnowledgeBot AI Agent

knowledgebot_instructions = """
Context:

You are an expert AI Tutor.

Instructions:
When given a question, provide a short factual answer based on your knowledge.

Input:
You will receive a question that you need to answer.

Output:
Respond with:
A consice and accurate answer to the question.If possible share the source of your information. If you don't know the answer, say "I don't know
"""

# Create a new agent called "Knowledge Bot"
knowledgebot_agent = Agent(name = "Knowledge Bot",   # Name of the agent
                           instructions = knowledgebot_instructions, # The rules and behavior for the agent
                           model = "gpt-4.1-mini") # The AI model (LLM) to use

# Print a confirmation message that the agent was created
print(f"Agent '{knowledgebot_agent.name}' created successfully!")

# A question that we need to answer.
session = SQLiteSession("conversation")

async def main(questiontext:str) -> str:
    # Replaced print_markdown with standard print

    # Run the Knowledge Bot agent on the input question]

    response = await Runner.run(    
        starting_agent = knowledgebot_agent,  # The agent we created earlier
        input = questiontext,
        session = session                    # The question we want it to answer
    )
    
    # Display the agent's response
    return response.final_output
    

    
# This kicks off the asynchronous event loop in a standard Python execution
if __name__ == "__main__":
    question1 = "What is the latest version of ChatGpt model that can be used for production application?"
    print("\n🤖 Agent's Response:\n")
    print(asyncio.run(main(question1)))
    
# Second question — depends on previous context
# Follow-up question that refers to the previous answer
if __name__ == "__main__":
    nextquestion = "How does performance compare to the previous version?"
    print("\n🤖 Agent's Response:\n")
    print(asyncio.run(main(nextquestion)))

