from asyncio.log import logger
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
import logging

import json
import modin.pandas as pd
import numpy as np

from emoji import demojize
from nltk.tokenize import TweetTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

tokenizer = TweetTokenizer()


def create_and_save_splits(data, dir_to_save_splits, train_dev_split=(80, 20)):
    assert len(train_dev_split) == 2
    assert sum(train_dev_split) == 100

    num_train = int((train_dev_split[0] / 100) * len(data))
    train, dev = np.split(data.sample(frac=1, random_state=42), [num_train])

    logger.info("***** Saving tweet texts to file *****")
    train_path = os.path.join(dir_to_save_splits, "train.txt")
    dev_path = os.path.join(dir_to_save_splits, "dev.txt")

    with open(train_path, "w") as f_w:
        for text in train["text"]:
            f_w.write(text + "\n")

    with open(dev_path, "w") as f_w:
        for text in dev["text"]:
            f_w.write(text + "\n")


def filter_text(text):
    token_len = len(tokenizer.tokenize(text))
    if token_len < 3:
        return True
    return False


def preprocess_tweet(tweet):
    tokens = tokenizer.tokenize(tweet)
    for i, token in enumerate(tokens):
        lowercased_token = token.lower()
        if lowercased_token.startswith("http") or lowercased_token.startswith("www"):
            tokens[i] = "[URL]"
        elif token.startswith("@"):
            tokens[i] = "[USER]"
        elif len(token) == 1:
            tokens[i] = demojize(token, delimiters=("", ""))
    return " ".join(tokens)


def get_all_tweet_texts(filepath):
    texts = []
    with open(filepath, "r") as f:
        for line in f:
            tweet_obj = json.loads(line)
            if "text" in tweet_obj:
                text = tweet_obj["text"]
                if not filter_text(text):
                    texts.append(preprocess_tweet(text))
    logger.info(f"done getting tweet texts from {filepath.name}")
    return texts


def main():
    month = "07"
    year = "2020"
    data1 = Path(
        f"/home/zqxh49/Development/phd/covid19-online-religion/data/twitter/UK/{year}/{month}/raw/replies/"
    )
    data2 = Path(
        f"/media/zqxh49/C28AAF378AAF273F/PHD/data/Covid-19/UK/{year}/{month}-{year}/"
    )
    output_dir = Path(
        f"/home/zqxh49/Development/phd/covid19-online-religion/data/twitter/UK/{year}/{month}/"
    )

    filepaths = []
    filepaths.append(data1.glob("*.jsonl"))
    filepaths.append(data2.glob(f"*-{month}-*.jsonl"))
    filepaths = list(chain(*filepaths))

    logger.info(f"***** Total no of filepaths: {len(filepaths)} *****")

    logger.info("***** Getting all tweet texts *****")
    with ProcessPoolExecutor() as executor:
        texts = executor.map(get_all_tweet_texts, filepaths)

    df = pd.DataFrame(list(chain(*texts)))
    df.columns = ["text"]

    # drop duplicates
    df = df[~df["text"].duplicated(keep="first")].reset_index(drop=True)

    # create and save splits to file
    create_and_save_splits(df, output_dir)

    # print("***** Saving tweet texts to file *****")
    # df.to_json(os.path.join(output_dir, "train.json"), orient="records", lines=True)


if __name__ == "__main__":
    main()
