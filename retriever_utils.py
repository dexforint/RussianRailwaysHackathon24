import os
from faiss_utils import search_in_index
import pickle
import numpy as np
from text_embedding import get_embedding
from llm_utils import generate_paraphrases_keywords_pseduoanswers_for_query

if os.path.exists("./data/chunk_db.pkl"):
    with open("./data/chunk_db.pkl", "rb") as f:
        chunk_db = pickle.load(f)
else:
    chunk_db = None


def get_relevant_docs(query: str, user_info: str | None = None limit: int = 2):
    query_data = generate_paraphrases_keywords_pseduoanswers_for_query(query)

    keywords = query_data["keywords"]
    keywords_vecs = []

    for keyword in keywords:
        keywords_vecs.append(get_embedding(keyword, task="query"))

    keywords_vecs = np.stack(keywords_vecs, axis=0)

    paraphrases = query_data["paraphrases"]
    paraphrases_query_vecs = [get_embedding(query, task="query")]
    # paraphrases_passage_vecs = [get_embedding(query, task="passage")]

    for paraphrase in paraphrases:
        paraphrases_query_vecs.append(get_embedding(paraphrase, task="query"))
        # paraphrases_passage_vecs.append(get_embedding(paraphrase, task="passage"))

    paraphrases_query_vecs = np.stack(paraphrases_query_vecs, axis=0)
    # paraphrases_passage_vecs = np.stack(paraphrases_passage_vecs, axis=0)

    pseudoanswers = query_data["pseudoanswers"]
    pseudoanswers_vecs = []

    for pseudoanswer in pseudoanswers:
        pseudoanswers_vecs.append(get_embedding(pseudoanswer, task="passage"))

    pseudoanswers_vecs = np.stack(pseudoanswers_vecs, axis=0)

    #####################

    keywords_doc_id2score = search_in_index("keywords", keywords_vecs)

    paraphrases_to_chunks_doc_id2score = search_in_index(
        "chunk", paraphrases_query_vecs
    )

    paraphrases_to_questions_doc_id2score = search_in_index(
        "questions", paraphrases_query_vecs
    )

    pseudoanswers_doc_id2score = search_in_index("chunk", pseudoanswers_vecs)

    chunk_id2score = unite_relevant_chunks(
        keywords_doc_id2score,
        paraphrases_to_chunks_doc_id2score,
        paraphrases_to_questions_doc_id2score,
        pseudoanswers_doc_id2score,
    )

    doc_id2score = {}
    for 

    relevant_docs = [chunk_db[doc_id] for doc_id in relevant_docs]
    relevant_docs = relevant_docs[:limit]

    return relevant_docs


def unite_relevant_chunks(
    keywords_relevant_docs,
    paraphrases_to_chunks_relevant_docs,
    paraphrases_to_questions_relevant_docs,
    pseudoanswers_relevant_docs,
):
    chunk_id2score = {}

    for doc_id, score in keywords_relevant_docs.items():
        chunk_id2score[doc_id] = chunk_id2score.get(doc_id, 0) + score * 1.0

    for doc_id, score in paraphrases_to_chunks_relevant_docs.items():
        chunk_id2score[doc_id] = chunk_id2score.get(doc_id, 0) + score * 1.0

    for doc_id, score in paraphrases_to_questions_relevant_docs.items():
        chunk_id2score[doc_id] = chunk_id2score.get(doc_id, 0) + score * 1.0

    for doc_id, score in pseudoanswers_relevant_docs.items():
        chunk_id2score[doc_id] = chunk_id2score.get(doc_id, 0) + score * 1.0


    # relevants_docs = list(doc_id2score.items())
    # relevants_docs.sort(key=lambda x: x[1], reverse=True)
    # relevants_docs = [doc_id for doc_id, score in relevants_docs]

    return chunk_id2score
