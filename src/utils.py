import re
import time
from openai import RateLimitError


def clean_json(response_text: str) -> str:
    """Strip markdown code blocks from LLM response before parsing JSON."""
    return re.sub(r'```json\n?|\n?```', '', response_text).strip()


def llm_call(client, model: str, input: list, max_retries: int = 5) -> str:
    """Call OpenAI API with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            response = client.responses.create(model=model, input=input)
            return response.output_text
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            print(f"Rate limit hit. Waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait_time)