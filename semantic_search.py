import os
import pandas as pd
import logging
import math
import argparse

from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

from utils.utils import init_logger

model = SentenceTransformer("all-mpnet-base-v2")

MAX_CORPUS_SIZE = 1000000

logger = logging.getLogger(__name__)
init_logger()


def read(fpath):
    dataf = pd.read_json(fpath, lines=True)
    dataf = dataf[["id", "text"]]
    return dataf


def get_cos_scores(query_embedding, corpus_embeddings):
    if torch.cuda.is_available():
        corpus_embeddings = corpus_embeddings.to("cuda")
        query_embedding = query_embedding.to("cuda")

    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    logger.info(f" Getting cosine similarity scores")
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    logger.info(f" FINISHED getting cosine similarity scores")

    return cos_scores


def main(args):
    logger.info(f" Reading query sentences from {args.query_path}")
    with open(args.query_path, "r") as f:
        queries = f.read().splitlines()
    query_embedding = model.encode(queries, convert_to_tensor=True)
    logger.info(f" Size of query sentences: {len(queries)}")

    # get average embedding for all queries
    query_embedding = torch.mean(query_embedding, dim=0)

    logger.info(f" Reading corpus from {args.input_path}")
    dataf = read(args.input_path)

    if len(dataf) > MAX_CORPUS_SIZE:
        logger.info(
            f" Size of corpus ({len(dataf)}) above limit ({MAX_CORPUS_SIZE}), splitting corpus into chunks of {MAX_CORPUS_SIZE}"
        )
        split_size = math.ceil(len(dataf) / MAX_CORPUS_SIZE)
        chunks = np.array_split(dataf, split_size)

        output = []
        for idx, chunk in enumerate(chunks):
            logger.info(f"chunk {idx + 1}")
            corpus_embeddings = model.encode(chunk["text"], convert_to_tensor=True)

            cos_scores = get_cos_scores(query_embedding, corpus_embeddings)

            output.extend(list(zip(chunk["id"], cos_scores)))
    else:
        corpus_embeddings = model.encode(dataf["text"], convert_to_tensor=True)
        logger.info(f" Size of corpus: {len(dataf)}")
        cos_scores = get_cos_scores(query_embedding, corpus_embeddings)
        output = list(zip(dataf["id"], cos_scores))

    logger.info(f" Writing cosine similarity scores to {args.output_path}")
    with open(args.output_path, "w") as f_w:
        for _id, cos in output:
            f_w.write(f"{_id},{cos.item():.4f}" + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--query_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    main(args)
