import os
import asyncio
from typing import List, Dict, Any
from agents import Agent, SQLiteSession, Runner

class SlidingWindowSession(SQLiteSession):
    """An extension of SQLiteSession that implements an active message capping strategy."""
    def __init__(self, session_id: str, max_messages: int = 6):
        super().__init__(session_id)
        self.max_messages = max_messages

    def get_messages(self) -> List[Dict[str, Any]]:
        """Retrieves messages from the database but clips the window to the latest messages."""
        raw_history = super().get_messages()
        
        # Always preserve the system instruction if it's stored at index 0
        if raw_history and raw_history[0].get("role") == "system":
            system_prompt = raw_history[0]
            conversational_turns = raw_history[1:]
            
            # Slice down to the most recent turns
            if len(conversational_turns) > self.max_messages:
                print(f"⚠️ [SLIDING WINDOW LOG]: Evicting {len(conversational_turns) - self.max_messages} old turns from active context.")
                conversational_turns = conversational_turns[-self.max_messages:]
            
            return [system_prompt] + conversational_turns
            
        return raw_history[-self.max_messages:]

# Usage inside your workshop environment:
session_sliding = SlidingWindowSession("long_thread_sliding", max_messages=4)



import os
from openai import OpenAI
from agents import SQLiteSession

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SummarizedMemorySession(SQLiteSession):
    """A memory session that condenses its history once it hits an operational threshold."""
    def __init__(self, session_id: str, trigger_threshold: int = 10):
        super().__init__(session_id)
        self.trigger_threshold = trigger_threshold
        self.summary_context = ""

    def get_messages(self) -> List[Dict[str, Any]]:
        raw_history = super().get_messages()
        
        # If history gets too deep, trigger an optimization compression cycle
        if len(raw_history) >= self.trigger_threshold:
            self._compress_and_summarize(raw_history)
            # Re-fetch the newly structured, clean history footprint
            raw_history = super().get_messages()
            
        return raw_history

    def _compress_and_summarize(self, history: List[Dict[str, Any]]):
        print("\n⚡ [SUMMARIZATION LOOP]: Compiling long history thread footprint...")
        
        # Format the chat history logs as text to be read by the summarizer
        formatted_transcript = ""
        for msg in history:
            if msg.get("role") in ["user", "assistant"]:
                formatted_transcript += f"{msg['role'].upper()}: {msg['content']}\n"
                
        summary_prompt = f"""
        Analyze the following agent chat history transcript. 
        Compress the transcript into a bulleted list of essential factual updates, core parameters found, 
        and resolved user data points. Omit chat pleasantries.
        
        TRANSCRIPT:
        {formatted_transcript}
        """
        
        # Trigger an independent context distillation call
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini", # Using a highly efficient model to keep costs down
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        self.summary_context = response.choices[0].message.content
        print(f"📝 [SUMMARIZATION COMPLETE]: New State Summary:\n{self.summary_context}\n")
        
        # Clear the database session data history logs safely
        self.clear() 
        
        # Re-inject the compiled historical facts straight into the top of the thread state
        self.save_message({
            "role": "system", 
            "content": f"Historical Session Context Summary:\n{self.summary_context}\n\nContinue current session tracking based on these historical facts."
        })


# Assuming you have a basic vector utility function available in your notebook environment
def search_archived_conversations_vector_db(query: str, thread_id: str) -> List[str]:
    """Simulates searching an external vector index repository for old chat segments."""
    # In practice, this converts the query to embeddings, runs a cosine similarity calculation,
    # and retrieves conversational fragments matching historical topic nodes.
    if "gemini" in query.lower() and "acme" in query.lower():
        return ["[ARCHIVED TURN 2026-05-12]: User verified active billing ID for Acme Corp. Tool reported 500 Gemini Enterprise licenses allocation."]
    return []

# Modifying your application runtime pipeline to support context injection
async def run_agent_with_archival_rag(user_prompt: str, session: SQLiteSession, agent_blueprint: Agent):
    print(f"👉 Processing Input: '{user_prompt}'")
    
    # 1. Query the conversational vector archive index first
    historical_fragments = search_archived_conversations_vector_db(
        query=user_prompt, 
        thread_id=session.session_id
    )
    
    # 2. Inject retrieved structural memory directly into the current execution prompt
    enriched_input = user_prompt
    if historical_fragments:
        print("📚 [ARCHIVAL MEMORY HIT]: Retrieved relevant context turn from vector store.")
        memory_block = "\n".join(historical_fragments)
        enriched_input = f"Relevant historical reference points:\n{memory_block}\n\nCurrent User Request: {user_prompt}"
    
    # 3. Fire runner pipeline execution loop safely
    response = await Runner.run(    
        starting_agent=agent_blueprint,
        input=enriched_input,
        session=session
    )
    return response.final_output