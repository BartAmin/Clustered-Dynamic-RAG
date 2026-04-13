import os
import time
import json
import yaml
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import datasets
from src.retrieval  import simple_retrieval, clustered_dynamic_retrieval
from src.generation import generate_answer, judge_answers

load_dotenv()

# ── Load config ───────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

METRICS = config['evaluation']['metrics']
K       = config['retrieval']['k']

# ── Load data and index ───────────────────────────────────────────────────
qa      = datasets.load_dataset("isaacus/legal-rag-bench", name="qa")
mapping = pd.read_pickle("results/mapping.pkl")
client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("results/cluster_keywords.json", "r") as f:
    cluster_keywords = json.load(f)

# ── Precompute ────────────────────────────────────────────────────────────
cluster_size   = mapping.groupby('cluster').size()
all_embeddings = np.vstack(mapping['embedding'].values)
all_ids        = mapping['id'].values

# ── Load embedding model for query encoding ───────────────────────────────
embed_model = SentenceTransformer(config['embedding_model'])

# ── Evaluation loop ───────────────────────────────────────────────────────
results = []

for i in range(len(qa['test']['answer'])):
    print(f"\nProcessing question {i+1}/{len(qa['test']['answer'])}...")

    question     = qa['test']['question'][i]
    ground_truth = qa['test']['answer'][i]
    query_embedding = embed_model.encode(question)

    # ── CDRAG ─────────────────────────────────────────────────────────────
    try:
        cdrag_context = clustered_dynamic_retrieval(
            question, query_embedding, mapping,
            cluster_keywords, cluster_size, K, config, client
        )
        result_cdrag = generate_answer(question, cdrag_context, config, client)
    except Exception as e:
        print(f"CDRAG failed for question {i}: {e}")
        continue

    # ── Simple RAG ────────────────────────────────────────────────────────
    try:
        simple_context = simple_retrieval(query_embedding, all_embeddings, all_ids, mapping, K)
        result_simple  = generate_answer(question, simple_context, config, client)
    except Exception as e:
        print(f"Simple RAG failed for question {i}: {e}")
        continue

    # ── LLM Judge ─────────────────────────────────────────────────────────
    try:
        cdrag, sim = judge_answers(
            question, ground_truth,
            result_cdrag, result_simple,
            config, client
        )
    except Exception as e:
        print(f"Judge failed for question {i}: {e}")
        continue

    # ── Store results ─────────────────────────────────────────────────────
    results.append({
        "question":      question,
        "ground_truth":  ground_truth,
        "result_cdrag":  result_cdrag,
        "result_simple": result_simple,
        "cdrag":         cdrag,
        "simple_rag":    sim
    })

    # ── Running averages ──────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"RUNNING AVERAGES after {len(results)} question(s):")
    print(f"{'Metric':<25} {'CDRAG':>15} {'Simple RAG':>15} {'Winner':>10}")
    print(f"{'─'*70}")
    for metric in METRICS:
        running_cdrag = np.mean([r['cdrag'][metric] for r in results])
        running_sim   = np.mean([r['simple_rag'][metric] for r in results])
        winner        = "CDRAG ✅" if running_cdrag > running_sim else ("Simple ✅" if running_sim > running_cdrag else "Tie 🟰")
        print(f"{metric:<25} {running_cdrag:>15.2f} {running_sim:>15.2f} {winner:>10}")

    time.sleep(2)

# ── Save results ──────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv("results/results.csv", index=False)

# ── Final summary ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("FINAL EVALUATION SUMMARY")
print(f"{'='*70}")
print(f"{'Metric':<25} {'CDRAG':>15} {'Simple RAG':>15} {'Winner':>10}")
print(f"{'-'*70}")
for metric in METRICS:
    avg_cdrag = results_df['cdrag'].apply(lambda x: x[metric]).mean()
    avg_sim   = results_df['simple_rag'].apply(lambda x: x[metric]).mean()
    winner    = "CDRAG ✅" if avg_cdrag > avg_sim else ("Simple ✅" if avg_sim > avg_cdrag else "Tie 🟰")
    print(f"{metric:<25} {avg_cdrag:>15.2f} {avg_sim:>15.2f} {winner:>10}")