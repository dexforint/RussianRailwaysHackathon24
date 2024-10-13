import os
from faiss_utils import search_in_index
import pickle
import numpy as np
from text_embedding import get_embedding
from llm_utils import generate_paraphrases_keywords_pseduoanswers_for_query

if os.path.exists("./data/chunks.pkl"):
    with open("./data/chunks.pkl", "rb") as f:
        chunk_db = pickle.load(f)
else:
    chunk_db = None


def get_relevant_docs(query: str, user_info: str | None = None, limit: int = 2):
    query_data = generate_paraphrases_keywords_pseduoanswers_for_query(query, user_info)

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

    #####################

    keywords_chunk_id2score = search_in_index("keywords", keywords_vecs)

    paraphrases_doc_id2score = search_in_index(
        "query_questions", paraphrases_query_vecs
    )

    summary_paraphrases_doc_id2score = search_in_index(
        "summary", paraphrases_query_vecs
    )

    # pseudoanswers_doc_id2score = search_in_index("chunk", pseudoanswers_vecs)

    chunk_id2score = unite_relevant_chunks(
        keywords_chunk_id2score,
        paraphrases_doc_id2score,
        summary_paraphrases_doc_id2score,
    )

    chunks = list(chunk_id2score.items())
    chunks.sort(key=lambda x: x[1], reverse=True)

    relevant_chunk_ids = [chunk[0] for chunk in chunks]

    relevant_chunks = [chunk_db[chunk_id] for chunk_id in relevant_chunk_ids]
    relevant_chunks = relevant_chunks[:limit]

    return relevant_chunks


def unite_relevant_chunks(
    keywords_chunk_id2score,
    paraphrases_doc_id2score,
    summary_paraphrases_doc_id2score,
):
    chunk_id2score = {}

    for chunk_id, score in keywords_chunk_id2score.items():
        chunk_id2score[chunk_id] = chunk_id2score.get(chunk_id, 0) + score * 1.0

    for chunk_id, score in paraphrases_doc_id2score.items():
        chunk_id2score[chunk_id] = chunk_id2score.get(chunk_id, 0) + score * 1.0

    for chunk_id, score in summary_paraphrases_doc_id2score.items():
        chunk_id2score[chunk_id] = chunk_id2score.get(chunk_id, 0) + score * 1.0

    # relevants_docs = list(doc_id2score.items())
    # relevants_docs.sort(key=lambda x: x[1], reverse=True)
    # relevants_docs = [doc_id for doc_id, score in relevants_docs]

    return chunk_id2score
