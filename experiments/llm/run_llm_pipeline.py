import time
import mlflow
from rag.retriever import FaissRetriever
from evaluation.llm_eval import simple_quality_score
import mlflow
mlflow.set_tracking_uri("file:./mlruns")

# Fake LLM call (replace with OpenAI or local LLM later)
def call_llm(prompt):
    time.sleep(0.5)
    return f"Generated answer for prompt: {prompt[:50]}..."

documents = [
    "MLflow is used for experiment tracking in ML systems.",
    "RAG improves LLM accuracy by retrieving relevant documents.",
    "Prompt engineering affects LLM output quality."
]

retriever = FaissRetriever(documents)

prompt_version = "v1"
prompt_path = f"prompts/{prompt_version}.txt"

with open(prompt_path) as f:
    prompt_template = f.read()

question = "What is MLflow and why is it used?"
prompt = prompt_template.format(question=question)

mlflow.set_experiment("llm-rag-experiments")

with mlflow.start_run():
    start = time.time()
    retrieved_docs = retriever.retrieve(question, top_k=3)
    response = call_llm(prompt + "\n".join(retrieved_docs))
    latency = (time.time() - start) * 1000

    quality = simple_quality_score(response)

    mlflow.log_param("prompt_version", prompt_version)
    mlflow.log_param("retriever", "faiss")
    mlflow.log_param("top_k", 3)

    mlflow.log_metric("latency_ms", latency)
    mlflow.log_metric("quality_score", quality)

    mlflow.log_text(prompt, "prompt.txt")
    mlflow.log_text(response, "response.txt")

print("LLM experiment logged to MLflow")