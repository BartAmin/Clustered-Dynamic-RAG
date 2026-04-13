import os
import json
import pickle
import yaml
from openai import OpenAI 
from dotenv import load_dotenv
import datasets
from src.index import build_index

load_dotenv()

# ── Load config ───────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ── Load data ─────────────────────────────────────────────────────────────
corpus = datasets.load_dataset("isaacus/legal-rag-bench", name="corpus")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Build index ───────────────────────────────────────────────────────────
mapping, cluster_keywords, embed_model = build_index(corpus, config, client)

# ── Save to disk (run once, reuse in evaluate.py) ─────────────────────────
mapping.to_pickle("results/mapping.pkl")

with open("results/cluster_keywords.json", "w") as f:
    json.dump(cluster_keywords, f, indent=2)

print("Index built and saved to results/")
print(f"  - Clusters: {mapping['cluster'].nunique()}")
print(f"  - Documents: {len(mapping)}")
print(f"  - Keywords: {list(cluster_keywords.items())[:2]}...")