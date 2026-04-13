import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import clean_json, llm_call


def simple_retrieval(query_embedding: np.ndarray, all_embeddings: np.ndarray,
                     all_ids: np.ndarray, mapping, k: int) -> str:
    """
    Simple RAG: retrieve top-k documents by cosine similarity
    across the entire corpus.
    """
    similarities  = cosine_similarity(query_embedding.reshape(1, -1), all_embeddings)[0]
    top_k_indices = np.argsort(-similarities)[:k]
    included_ids  = all_ids[top_k_indices].tolist()
    context       = " ".join(mapping.loc[mapping['id'].isin(included_ids), 'text'])
    return context


def clustered_dynamic_retrieval(question: str, query_embedding: np.ndarray,
                                mapping, cluster_keywords: dict,
                                cluster_size, k: int, config: dict, client) -> str:
    """
    CDRAG: LLM selects relevant clusters,
    then retrieves top sub_k documents per cluster by cosine similarity.
    """

    # ── Step 1: Ask LLM which clusters to retrieve from ───────────────────
    retrieval_system_prompt = f"""
        You are a legal document retrieval assistant. Given a question and a set of document clusters, 
        select the most relevant documents to answer the question.

        Each cluster is represented by keywords indicating its content:
        {cluster_keywords}

        Each cluster contains {cluster_size} documents.

        Task:
        - Retrieve up to {k} documents total across clusters.
        - Assign sub_k documents per cluster group, where sub_k is the number of documents to retrieve.
        - If a relevant keyword appears in multiple clusters, retrieve sub_k documents from all those clusters.

        Constraints:
        - Sum of all sub_k values must equal {k}.
        - sub_k for any cluster cannot exceed {cluster_size}.

        Output ONLY raw JSON (no markdown, no code blocks, no explanation):
        {{
            "retrievals": [
                {{"sub_k": <int>, "clusters": ["cluster_x", "cluster_y"]}},
                {{"sub_k": <int>, "clusters": ["cluster_z"]}},
                ...
            ]
        }}

        IMPORTANT: Your response must be valid JSON only. 
        Do not include any text before or after the JSON.
        Do not use trailing commas.
        Do not add comments.
    """

    response_text = llm_call(
        client,
        model=config['llm_model'],
        input=[
            {"role": "system", "content": retrieval_system_prompt},
            {"role": "user",   "content": question}
        ]
    )
    retrieval_result = json.loads(clean_json(response_text))

    # ── Step 2: Retrieve top documents per cluster ────────────────────────
    included_ids = []

    for docs in retrieval_result['retrievals']:
        clusters = docs['clusters']
        sub_k    = docs['sub_k']

        # Filter mapping for relevant clusters, excluding already retrieved docs
        cluster_selection = mapping.loc[
            mapping['cluster'].isin(clusters) & ~mapping['id'].isin(included_ids),
            ['id', 'embedding']
        ]

        if cluster_selection.empty:
            continue

        # Compute cosine similarity and retrieve top sub_k docs
        cluster_embeddings = np.vstack(cluster_selection['embedding'].values)
        similarities       = cosine_similarity(query_embedding.reshape(1, -1), cluster_embeddings)[0]
        top_k_indices      = np.argsort(-similarities)[:sub_k]
        top_k_ids          = cluster_selection['id'].values[top_k_indices]
        included_ids.extend(top_k_ids)

    context = " ".join(mapping.loc[mapping['id'].isin(included_ids), 'text'])
    return context