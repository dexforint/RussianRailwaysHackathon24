from llm_tools import get_llm_response, get_json_from_text


def get_prompt(prompt_name: str, **kwargs) -> str:
    with open(f"prompts/{prompt_name}.txt", "r") as f:
        prompt = f.read()
        return prompt.format(**kwargs)


def get_llm_response(prompt: str):
    pass


def generate_answer(query: str):
    relevant_docs = get_relevant_docs(query)

    relevant_docs_str = []
    sources = []
    for i, doc in enumerate(relevant_docs):
        # doc["title"]
        # doc["url"]
        sources.append(
            {
                "title": doc["title"],
                "url": doc["url"],
            }
        )
        relevant_docs_str.append(f"Документ №{i}\n{doc["text"]}")

    relevant_docs_str = "\n=======================\n".join(relevant_docs_str)

    prompt = get_prompt("main_prompt", query=query, relevant_docs_str=relevant_docs_str)

    response = get_llm_response(prompt)
    data = get_json_from_text(response)

    answer = postprocess_answer(data, sources)

    return answer


def postprocess_answer(data: dict, sources: list):
    doc_indexes = data["docs"]
    answer = data["answer"]


def get_relevant_docs(query: str):
    pass


def generate_summary_questions_keywords_for_text(text: str, context: str):
    prompt = get_prompt("prompt_for_chunk")

    response = get_llm_response(prompt)

    data = get_json_from_text(response)

    return data


def generate_paraphrases_keywords_pseduoanswers_for_query(query: str, context: str):
    prompt = get_prompt("prompt_for_query")

    response = get_llm_response(prompt)

    data = get_json_from_text(response)

    return data
