import os
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

# Initialize the OpenAI client (it automatically looks for OPENAI_API_KEY env variable)
client = OpenAI()

# Define the structure of a single action item
class ActionItem(BaseModel):
    task: str = Field(description="The specific action that needs to be taken.")
    deadline: str = Field(description="The mentioned deadline, or 'Not specified'.")
    priority: str = Field(description="High, Medium, or Low based on urgency.")
    assignee: str = Field(description="Who needs to do this. Default to 'User' if unclear.")

# Define the final response layout for the agent
class DigestReport(BaseModel):
    summary: str = Field(description="A brief 2-sentence summary of the overall situation.")
    action_items: List[ActionItem] = Field(description="A list of extracted actionable tasks.")

def run_digest_agent(raw_text_input: str) -> DigestReport:
    # The agent's instructions (System Prompt)
    system_instruction = (
        "You are an elite Executive Assistant Agent. Your job is to analyze chaotic project updates, "
        "meeting notes, or emails, filter out the noise, and extract concrete action items. "
        "Be aggressive about finding hidden tasks that people agreed to do."
    )

    # Calling the OpenAI API using Structured Outputs
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",  # Highly cost-effective and smart enough for agentic tasks
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Please process this text:\n\n{raw_text_input}"}
        ],
        response_format=DigestReport, # Forces the agent to reply in our exact schema
    )

    return completion.choices[0].message.parsed

# --- Test Data to simulate a chaotic workday input ---
sample_dump = """
Hey team, quick update from the morning sync. Dave, we absolutely need the Q3 budget slides 
ready by Friday morning because Sarah is reviewing them with investors. Speaking of Sarah, she 
mentioned she likes the new logo direction, but someone needs to send her the high-res PNGs by tonight. 
I'll handle booking the conference room for next week. Also, don't forget we need to update the 
readme file on GitHub eventually, maybe next month.
"""

if __name__ == "__main__":
    print("🤖 Agent is analyzing the data...\n")
    report = run_digest_agent(sample_dump)
    
    print(f"📋 SUMMARY:\n{report.summary}\n")
    print("⚡ ACTION ITEMS EXTRACTED:")
    for item in report.action_items:
        print(f"- [{item.priority} Priority] {item.task} (Assignee: {item.assignee}, Due: {item.deadline})")