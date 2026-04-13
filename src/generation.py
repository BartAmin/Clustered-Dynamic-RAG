import json
from src.utils import clean_json, llm_call


def generate_answer(question: str, context: str, config: dict, client) -> str:
    """Generate a concise legal answer based on the provided context."""

    system_prompt = f"""
        You are an expert legal assistant. Answer the question concisely and accurately 
        based solely on the provided context.

        Guidelines:
        * Base your answer strictly on the provided context.
        * Limit your answer to 50 words +- 20%.
        * If the context is insufficient, state: 
          "The provided context does not contain sufficient information to answer this question."
        * Do not make assumptions or use knowledge outside the provided context.

        Context:
        {context}
    """

    return llm_call(
        client,
        model=config['llm_model'],
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question}
        ]
    )


def judge_answers(question: str, ground_truth: str,
                  result_cdrag: str, result_simple: str,
                  config: dict, client) -> tuple[dict, dict]:
    """
    LLM judge evaluates both answers against the ground truth.
    Returns scores for CDRAG and simple RAG.
    """

    judge_system_prompt = """
        You are an expert legal AI evaluator. Evaluate two RAG system answers (System A and System B) 
        to a legal question based on the provided ground truth answer.

        Score each answer on the following metrics (1-5):
        1. Faithfulness: Is the answer factually consistent with the ground truth?
        2. Answer Relevance: Does the answer directly address the question?
        3. Completeness: Does the answer cover all key aspects of the ground truth?
        4. Conciseness: Is the answer free of irrelevant information?
        5. Semantic Similarity: How semantically close is the answer to the ground truth?
        6. Overall: Overall quality score.

        Output ONLY raw JSON (no markdown, no code blocks, no explanation):
        {
            "system_a": {
                "faithfulness": <1-5>,
                "answer_relevance": <1-5>,
                "completeness": <1-5>,
                "conciseness": <1-5>,
                "semantic_similarity": <1-5>,
                "overall": <1-5>,
                "reasoning": "<brief explanation>"
            },
            "system_b": {
                "faithfulness": <1-5>,
                "answer_relevance": <1-5>,
                "completeness": <1-5>,
                "conciseness": <1-5>,
                "semantic_similarity": <1-5>,
                "overall": <1-5>,
                "reasoning": "<brief explanation>"
            }
        }
    """

    judge_user_prompt = f"""
        Question: {question}
        Ground Truth: {ground_truth}
        System A Answer: {result_cdrag}
        System B Answer: {result_simple}
    """

    response_text = llm_call(
        client,
        model=config['judge_model'],
        input=[
            {"role": "system", "content": judge_system_prompt},
            {"role": "user",   "content": judge_user_prompt}
        ]
    )

    judge_result = json.loads(clean_json(response_text))

    # system_a = CDRAG, system_b = simple RAG
    return judge_result['system_a'], judge_result['system_b']