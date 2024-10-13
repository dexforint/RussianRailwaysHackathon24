# !pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "jinaai/jina-embeddings-v3",
    trust_remote_code=True,
    device="cuda",
    cache_folder="./tmp",
)


def get_embedding(text: str, task="query"):
    task = f"retrieval.{task}"
    embeddings = model.encode(
        [text],
        task=task,
        prompt_name=task,
        use_flash_attn=False,
        show_progress_bar=False,
    )

    return embeddings[0]


def get_embeddings(texts: list[str], task="query"):
    task = f"retrieval.{task}"
    embeddings = model.encode(
        texts,
        task=task,
        prompt_name=task,
        use_flash_attn=False,
        show_progress_bar=False,
    )

    return embeddings


if __name__ == "__main__":
    import time

    emb = get_embedding("hello world")

    print(emb.shape)

    time.sleep(10)
