import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import fcluster, linkage
from openai import RateLimitError
from keybert import KeyBERT
from src.utils import clean_json, llm_call
import time

def build_index(corpus, config, client) -> tuple[pd.DataFrame, dict, SentenceTransformer]:
    """
    Full index building pipeline:
    1. Embed documents
    2. Cluster embeddings
    3. Extract keywords per cluster
    Returns mapping DataFrame and cluster_keywords dict.
    """

    # ── Step 1: Embed documents ───────────────────────────────────────────
    embed_model = SentenceTransformer(config['embedding_model'])
    texts       = list(corpus['test']['text'])
    embeddings  = embed_model.encode(texts)

    # ── Step 2: Hierarchical clustering ───────────────────────────────────
    linked         = linkage(embeddings, method='ward')
    cluster_labels = fcluster(
        linked,
        t=config['clustering']['threshold'],
        criterion=config['clustering']['criterion']
    )

    # t = 5.5 is not the only plausible value, but it results in more and smaller clusters, which gives the 
    # llm more choice in the dynamic retrieval part. To inspect the hierarchical structure, run the line below
    #dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)

    mapping = pd.DataFrame({
        "id":        corpus['test']['id'],
        "text":      texts,
        "embedding": [emb for emb in embeddings],
        "cluster":   [f"cluster_{label}" for label in cluster_labels]
    })

    # ── Step 3: Extract keywords per cluster chunks using LLM ──────────────
    system_prompt = f"""
        You are a keyword extraction expert. Extract up to 5 important keywords 
        from the given document. You will also receive a list of existing keywords - do not suggest keywords already in this list.
        Fewer keywords is acceptable if the document does not warrant more.
        Strict rule: Only output keywords, nothing more
    """

    cluster_keywords = {}
    chunk_size = 50000
    counter = 0

    for cluster in set(mapping['cluster']):

        # Extract only docs from selected cluster and concatenate them
        cluster_texts = mapping.loc[mapping['cluster'] == cluster, 'text']
        concatenated_text  = " ".join(cluster_texts.tolist())
        
        cluster_keywords[cluster] = []
        
        for i in range(0, len(concatenated_text), chunk_size):
            chunk = concatenated_text[i:i+chunk_size]

            if not chunk.strip():  
                continue
            
            # Ensure no keywords are wasted on ones that already exist in the list
            text_user_prompt = f"""
                Text: {chunk}
                Existing keywords: {cluster_keywords.get(cluster)}
            """

            for attempt in range(5):
                try:
                    response = client.responses.create(
                        model=config['llm_model'],
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": text_user_prompt}
                        ]
                    )
                    break
                except RateLimitError:
                    wait_time = 2 ** attempt
                    print(f"Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)

            if response.output_text.strip():
                cluster_keywords[cluster].append(response.output_text.strip())
            
        counter += 1
        print(f"keywords extracted from {counter} of {len(set(mapping['cluster']))} clusters")

    return mapping, cluster_keywords, embed_model