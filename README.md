# AgenticAI

Code for **Agentic AI Training** — a collection of hands-on Python modules covering structured outputs, retrieval-augmented generation (RAG), memory, tool use, guardrails, evaluation, and a capstone agentic application. This repository is used as the practical companion to the Agentic AI training delivered by **Technology Reboot**.

## Overview

The repo is organized as a sequence of training modules, each demonstrating a specific building block of agentic AI systems, from simple structured-output agents to multi-tool, memory-aware agents with guardrails and a full capstone use case.

## Repository Structure

```
AgenticAI/
├── Capstone/                                          # Capstone project assets
├── chroma_db/                                         # Local ChromaDB vector store (used by RAG modules)
├── ebooks/                                             # Source documents used for RAG ingestion
├── Compression.py                                      # Prompt/context compression example
├── G-Eval.py                                           # LLM-based evaluation (G-Eval) example
├── Module2.py                                          # Structured-output agent (executive assistant / action-item extractor)
├── Module2.1.py                                        # Module 2 follow-up exercise
├── Module2.2.py                                        # Module 2 follow-up exercise
├── Module2.3 - mem.py                                  # Adding memory to an agent
├── Module2.4 - mem-tools.py                            # Memory + tool-calling agent
├── Module2.5 - mem - guardrail.py                      # Memory + tool-calling agent with guardrails
├── Module 3 - RAG with Langchain.py                    # RAG pipeline using LangChain
├── Module 3 - RAG with Langchain-Local.py               # RAG pipeline using LangChain with a local model
├── Module 3 - RAG with Langchain-Local.-obsevabilitypy.py  # Local RAG pipeline with observability/tracing
├── Module 5 - Capstone Project.py                       # Capstone agentic application
├── Module 5 - Leave Management System.py                # Capstone-style applied example: agentic leave management
├── agent1_lead_sourcer.py                               # Agent example: automated lead sourcing
├── onedrive_connect.py                                  # Utility: connecting an agent/tool to OneDrive
└── README.md
```

> Note: file names mirror the order in which topics are taught during the training; module numbers are not strictly sequential by filename order in GitHub's default sort.

## What Each Module Covers

- **Module 2 series** — Building agents with structured outputs (Pydantic schemas via OpenAI's API), then progressively adding memory, tool calling, and guardrails.
- **Module 3 series** — Retrieval-Augmented Generation (RAG) using LangChain, including a cloud-model version, a local-model version, and a version instrumented for observability.
- **Module 5 series** — Capstone-style applied projects, including a full capstone project and an agentic Leave Management System example.
- **Compression.py** — Techniques for compressing prompts/context to reduce token usage.
- **G-Eval.py** — Using an LLM as an evaluator (G-Eval pattern) to score agent or model outputs.
- **agent1_lead_sourcer.py** — A standalone agent example for sourcing leads.
- **onedrive_connect.py** — Helper script for integrating an agent with OneDrive as a data/tool source.
- **chroma_db/** — Local vector store used by the RAG modules.
- **ebooks/** — Sample documents used as the knowledge base for RAG examples.
- **Capstone/** — A full multi-agent WAM (Wealth and Asset Management) Agentic Platform built with LangGraph: a lead-sourcing agent, an investment-suggestion agent, and a portfolio-diversification agent, orchestrated as a stateful graph and exposed via a FastAPI REST API. See Capstone/README.md for full details.

## Prerequisites

- Python 3.10+
- An OpenAI API key (set as the `OPENAI_API_KEY` environment variable), used by the structured-output and several agentic modules
- For the RAG modules: a LangChain-compatible setup, and (for the local-model variants) a local LLM runtime

## Installation

```bash
git clone https://github.com/technology-reboot/AgenticAI.git
cd AgenticAI
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt   # if present; otherwise install per-module dependencies (see below)
```

Since each module is largely self-contained, install only what a given script needs, for example:

```bash
pip install openai pydantic
pip install langchain langchain-openai chromadb
```

## Usage

Set your API key:

```bash
export OPENAI_API_KEY="your-api-key-here"   # On Windows: set OPENAI_API_KEY=your-api-key-here
```

Run any module directly, for example:

```bash
python Module2.py
```

For the RAG modules, ensure the `ebooks/` folder is populated with source documents and `chroma_db/` is writable, then run:

```bash
python "Module 3 - RAG with Langchain.py"
```

For the capstone project:

```bash
python "Module 5 - Capstone Project.py"
```

## About

This repository accompanies the **Agentic AI Training** delivered by [Technology Reboot](https://github.com/technology-reboot), Shailesh Pardesi's independent AI consulting and training practice, covering practical, applied agentic AI development for enterprise audiences.

## License

No license has been specified for this repository. Please contact the repository owner before reusing this code outside of the training context.
