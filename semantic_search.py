import os
import logging
from pathlib import Path
import json

from sentence_transformers import SentenceTransformer, util
import torch

from utils.utils import init_logger

model = SentenceTransformer("all-mpnet-base-v2")

MAX_CORPUS_SIZE = 1000000

logger = logging.getLogger(__name__)
init_logger()


def read(fpath):
    texts = []
    i = 0
    with open(fpath, "r") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            texts.append(text)
            i += 1
            if i == 100:
                break
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
    label = "islam"
    input_path = f"./data/reddit/processed/{label}.json"
    query_path = (
        f"./data/reddit/interim/queries_{label}.txt"
        if label == "islam"
        else "./data/reddit/interim/queries.txt"
    )

    output_path = f"./data/reddit/interim/cos_scores_{label}.txt"

    logger.info(f" Reading query sentences from {query_path}")
    with open(query_path, "r") as f:
        queries = f.read().splitlines()
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
        for idx, chunk in enumerate(chunks):
            logger.info(f"chunk {idx + 1}")
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
