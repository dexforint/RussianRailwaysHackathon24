from llm_utils import get_llm_response, get_prompt
import json
from retriever_utils import get_relevant_docs
from parse_json_tools import get_json_from_text
import os
import pickle


def generate_answer(query: str, user_info: str | None = None):
    relevant_docs = get_relevant_docs(query, user_info)

    relevant_docs_str = []
    for i, doc in enumerate(relevant_docs):
        relevant_docs_str.append(f"Документ №{i}\n{doc["text"]}")

    relevant_docs_str = "\n=======================\n".join(relevant_docs_str)

    prompt = get_prompt(
        "main_prompt",
        query=query,
        relevant_docs_str=relevant_docs_str,
        context=user_info,
    )

    response = get_llm_response(prompt)
    data = get_json_from_text(response)

    if len(data["docs"]) == 0:
        if os.path.exists("./data/unanswered_queries.pkl"):
            with open("./data/unanswered_queries.pkl", "rb") as f:
                unanswered_queries = pickle.load(f)
        else:
            unanswered_queries = []

        unanswered_queries.append(query)

        with open("./data/unanswered_queries.pkl", "wb") as f:
            pickle.dump(unanswered_queries, f)

    answer = postprocess_answer(data, relevant_docs)

    return answer


def postprocess_answer(data: dict, relevant_docs: list):
    doc_indexes = data["docs"]
    answer = data["answer"]

    sources_str = []

    for i, doc_index in enumerate(doc_indexes):
        source = relevant_docs[doc_index]
        sources_str.append(f'{i+1}. <a href="{source["url"]}">{source["title"]}</a>')

    sources_str = "<br />".join(sources_str)

    return f"{answer}<br /><br />Источники:<br />{sources_str}"


# def generate_paraphrases_keywords_pseduoanswers_for_query(query: str):
#     prompt = get_prompt("prompt_for_query", query=query)

#     response = get_llm_response(prompt)

#     data = get_json_from_text(response)
#     # data["query"] = query

#     return data


def evaluate_answer(pred_answer: str, gt_answer: str) -> float:
    prompt = get_prompt("evaluate_prompt", pred_answer=pred_answer, gt_answer=gt_answer)
    response = get_llm_response(prompt)

    data = get_json_from_text(response)

    score = data["score"]

    return score


def evaluate():
    with open("data/eval_data.json", "r") as f:
        eval_ds = json.load(f)

    scores = []

    for question, gt_answer in eval_ds:
        pred_answer = generate_answer(question)

        score = evaluate_answer(pred_answer, gt_answer)

        scores.append(score)

    mean_score = sum(scores) / len(scores)

    return mean_score


if __name__ == "__main__":
    answer = generate_answer("как определяется размер среднего заработка?")
    print(answer)
