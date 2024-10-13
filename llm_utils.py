"""Модель для взаимодействия с LLM"""

import requests
from parse_json_tools import parse_json as get_json_from_text


def get_llm_response(prompt: str) -> str:
    """Функция для получения ответа от LLM с помощью бесплатного API

    Args:
        prompt (str): промпт

    Returns:
        str: ответ нейронной сети
    """
    headers = {
        "accept": "application/json",
        "content-type": "application/x-www-form-urlencoded",
    }

    params = {
        "model_id": 23,
        "prompt": prompt,
    }

    response = requests.post(
        "https://api.qewertyy.dev/models", params=params, headers=headers
    )

    return response.json()["content"][0]["text"]


def get_prompt(prompt_name: str, **kwargs) -> str:
    with open(f"prompts/{prompt_name}.txt", "r", encoding="utf-8") as f:
        prompt = f.read()
        return prompt.format(**kwargs)


def generate_summary_questions_keywords_for_chunk(chunk: str, context: str):
    prompt = get_prompt("prompt_for_chunk", chunk=chunk, context=context)

    response = get_llm_response(prompt)

    data = get_json_from_text(response)

    return data


def generate_paraphrases_keywords_pseduoanswers_for_query(query: str):
    prompt = get_prompt("prompt_for_query", query=query)

    response = get_llm_response(prompt)

    data = get_json_from_text(response)
    # data["query"] = query

    return data
