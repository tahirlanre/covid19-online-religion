import os
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("all-mpnet-base-v2")

MAX_CORPUS_SIZE = 10

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def read(fpath):
    texts = []
    with open(fpath, "r") as f:
        texts = f.read().splitlines()
    return texts


def get_cos_scores(query_embedding, corpus_embeddings):
    if torch.cuda.is_available():
        corpus_embeddings = corpus_embeddings.to("cuda")
        query_embedding = query_embedding.to("cuda")

    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    logger.info(f" Getting cosine similarity scores")
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    logger.info(f" FINISHED getting cosine similarity scores")

    return cos_scores


def main():
    label = "sample"
    input_path = f"./data/reddit/keys_{label}.txt"
    query_path = (
        f"./data/reddit/queries_{label}.txt"
        if label == "islam"
        else "./data/reddit/queries.txt"
    )

    output_path = f"./data/reddit/cos_scores_{label}2.txt"

    logger.info(f" Reading query sentences from {query_path}")
    queries = read(query_path)
    query_embedding = model.encode(queries, convert_to_tensor=True)
    logger.info(f" Size of query sentences: {len(queries)}")

    # get average embedding for all queries
    query_embedding = torch.mean(query_embedding, dim=0)

    logger.info(f" Reading corpus from {input_path}")
    corpus = read(input_path)

    if len(corpus) > MAX_CORPUS_SIZE:
        logger.info(
            f" Size of corpus ({len(corpus)}) above limit ({MAX_CORPUS_SIZE}), splitting corpus into chunks of {MAX_CORPUS_SIZE}"
        )
        chunks = [
            corpus[i : i + MAX_CORPUS_SIZE]
            for i in range(0, len(corpus), MAX_CORPUS_SIZE)
        ]

        cos_scores = []
        for chunk in chunks:
            corpus_embeddings = model.encode(chunk, convert_to_tensor=True)

            _cos_scores = get_cos_scores(query_embedding, corpus_embeddings)
            cos_scores.extend(_cos_scores)
    else:
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
        logger.info(f" Size of corpus: {len(corpus)}")
        cos_scores = get_cos_scores(query_embedding, corpus_embeddings)

    logger.info(f" Writing cosine similarity scores to {output_path}")
    with open(output_path, "w") as f_w:
        for idx, score in enumerate(cos_scores):
            f_w.write(f"{idx},{score.item():.4f}" + "\n")


if __name__ == "__main__":
    main()
