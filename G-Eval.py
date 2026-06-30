import os
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# 1. Define the exact structured schema the Judge must adhere to
class FaithfulnessAssessment(BaseModel):
    rationales: List[str] = Field(
        description="Step-by-step logical critique analyzing whether each fact in the answer exists in the context."
    )
    contains_hallucination: bool = Field(
        description="True if any statement in the answer cannot be supported by the provided context. False otherwise."
    )
    score: float = Field(
        description="A continuous score between 0.0 (completely hallucinated) and 1.0 (perfectly faithful)."
    )

# 2. Construct the Systematic G-Eval Prompt
judge_prompt = ChatPromptTemplate.from_template("""
You are a highly rigorous, objective QA Evaluation Judge. Your task is to assess whether a generated answer is **faithful** to the provided context or if it contains **hallucinations**.

Evaluation Criteria:
- Faithfulness measures if the generated answer is entirely derived from the context.
- If the answer contains facts, statistics, or logical leaps not explicitly found in the context, it MUST be flagged as a hallucination.

Evaluation Steps:
1. Read the retrieved context thoroughly.
2. Break down the generated answer into discrete atomic factual statements.
3. For each atomic statement, check if it is directly stated or logically implied by the context.
4. Record your step-by-step reasoning for each statement in the 'rationales' array.
5. Provide a final boolean verdict and a numeric score.

[CONTEXT]
{context}

[GENERATED ANSWER]
{answer}

{format_instructions}
""")

def evaluate_run(context_str: str, generated_answer_str: str) -> FaithfulnessAssessment:
    # Use a high-tier reasoning model for evaluation
    judge_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    parser = PydanticOutputParser(pydantic_object=FaithfulnessAssessment)
    
    # Compile the evaluation chain
    eval_chain = judge_prompt | judge_llm | parser
    
    # Execute the evaluation
    result = eval_chain.invoke({
        "context": context_str,
        "answer": generated_answer_str,
        "format_instructions": parser.get_format_instructions()
    })
    return result

# --- LIVE TESTING DEMO ---
if __name__ == "__main__":
    sample_context = """
    The Alpha-9 microservice cluster is hosted natively on AWS ECS. It connects exclusively 
    to a PostgreSQL database instances via private VPC endpoints to maintain encryption at rest.
    """
    
    # This answer contains a hallucination (Google Cloud Platform)
    hallucinated_answer = """
    The Alpha-9 cluster operates on Google Cloud Platform and stores its operational data 
    inside a secure PostgreSQL database.
    """
    
    print("🔬 Running Systematic Evaluation...")
    evaluation_result = evaluate_run(sample_context, hallucinated_answer)
    
    print(f"\nIs Hallucinated: {evaluation_result.contains_hallucination}")
    print(f"Faithfulness Score: {evaluation_result.score}/1.0")
    print("\nJudge's Rationales:")
    for rationale in evaluation_result.rationales:
        print(f"- {rationale}")