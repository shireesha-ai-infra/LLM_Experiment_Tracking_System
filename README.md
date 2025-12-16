# ML & LLM Experiment Tracking System with MLflow

This project demonstrates a production-style experiment tracking system
for both ML models and LLM pipelines.

## Features
- Neural network experiment tracking
- Prompt versioning
- RAG configuration tracking
- Latency, quality, and cost-style metrics
- Artifact logging
- Visual comparison in MLflow UI

## Run
1. Start MLflow server
2. Run ML experiments
3. Run LLM + RAG experiments
4. Compare runs in MLflow UI

## Tech Stack
MLflow, PyTorch, FAISS, SentenceTransformers


> Always execute scripts from the project root using `python -m` to ensure correct module resolution.

Here, all scripts must be executed from the project root using:
python -m <module.path>