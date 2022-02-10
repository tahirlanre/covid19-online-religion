import logging

from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("all-mpnet-base-v2")

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


def main():
    input_path = "./data/reddit/keys_sample.txt"
    query_path = "./data/reddit/sentences_islam.txt"

    output_path = "./data/reddit/cos_scores_sample.txt"

    logger.info(f" Reading corpus from {input_path}")
    corpus = read(input_path)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    logger.info(f" Size of corpus: {len(corpus)}")

    logger.info(f" Reading query sentences from {query_path}")
    queries = read(query_path)
    query_embedding = model.encode(queries, convert_to_tensor=True)
    logger.info(f" Size of query sentences: {len(queries)}")

    # get average embedding for all queries
    query_embedding = torch.mean(query_embedding, dim=0)

    if torch.cuda.is_available():
        corpus_embeddings = corpus_embeddings.to("cuda")
        query_embedding = query_embedding.to("cuda")

    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    logger.info(f" Getting cosine similarity scores")
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    logger.info(f" FINISHED getting cosine similarity scores")
    
    logger.info(f" Writing cosine similarity scores to {output_path}")
    with open(output_path, "w") as f_w:
        for idx, score in enumerate(cos_scores):
            f_w.write(f"{idx},{score.item()}" + "\n")


if __name__ == "__main__":
    main()
