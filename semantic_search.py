from lib2to3.pytree import convert
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("all-mpnet-base-v2")


def read(fpath):
    texts = []
    with open(fpath, "r") as f:
        texts = f.read().splitlines()
    return texts


def main():
    input_path = "./data/reddit/keys_islam.txt"
    query_path = "./data/reddit/sentences_islam.txt"

    output_path = "./data/reddit/cos_scores_islam.txt"

    corpus = read(input_path)
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

    queries = read(query_path)
    query_embedding = model.encode(queries, convert_to_tensor=True)

    # get average embedding for all queries
    query_embedding = torch.mean(query_embedding, dim=0)

    if torch.cuda.is_available():
        corpus_embeddings = corpus_embeddings.to("cuda")
        query_embedding = query_embedding.to("cuda")

    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    for idx, score in cos_scores:
        with open(output_path, "w") as f_w:
            f_w.write(f"{idx},{score}" + "\n")


if __name__ == "__main__":
    main()
