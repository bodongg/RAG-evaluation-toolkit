# rag-eval-toolkit

A Python toolkit for evaluating the quality, speed, and cost of RAG systems.

## What it measures

- Retrieval accuracy: how well the system finds relevant documents.
- Faithfulness (hallucination rate): how well answers are supported by retrieved documents.
- Answer relevance: how well answers address the question.
- Latency and cost: how long answers take and how much they cost.

## Status

Day 5: vector store

## Planned features

- [ ] Document ingestion and chunking
- [ ] Baseline RAG pipeline
- [ ] Test set generation
- [ ] Metrics for retrieval accuracy, faithfulness, answer relevance, latency, and cost
- [ ] Config comparison for chunk size, reranker, and hybrid search
- [ ] Streamlit dashboard

## Setup

Run these commands from the project folder in PowerShell on Windows.

1. Create a virtual environment:

   ```powershell
   python -m venv venv
   ```

2. Activate it:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Copy the config template:

   ```powershell
   Copy-Item .env.example .env
   ```

5. Open `.env` and fill in your API key and model name. Adjust the other values as needed.
