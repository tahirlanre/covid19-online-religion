import logging
from time import time
import json
import os

import pandas as pd

from utils.utils import init_logger
from preprocess.preprocess import normalize_reddit_text

logger = logging.getLogger(__name__)
init_logger()


def process_data(path):
    """Read json data from file"""
    data = []
    with open(path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                dt = pd.to_datetime(rec["created_date"])
                text = rec["text"]
                # select only posts beween 2011 - 2020
                if dt.year >= 2011 and dt.year < 2021:
                    rec["text"] = normalize_reddit_text(text)
                    data.append(rec)
            except json.decoder.JSONDecodeError:
                logger.error("Error reading json line")
                continue
    return data


def write_texts(texts, path):
    with open(path, "w") as f_w:
        for text in texts:
            if text.strip():
                f_w.write(text + "\n")


def main():
    data_dir = "data/reddit/subreddit/"
    subreddits = ["yoga"]
    for subreddit in subreddits:
        logger.info(f"Processing subreddit: {subreddit}")
        start_time = time()
        json_file = os.path.join(data_dir, subreddit, "post.json")
        processed_data = process_data(json_file)

        df = pd.DataFrame(processed_data)
        df.drop_duplicates(subset=["id"], inplace=True)  # drop duplicate posts

        output_path = os.path.join(
            data_dir, subreddit, f"{subreddit.lower()}_texts.txt"
        )
        write_texts(df["text"], output_path)

        logger.info(f"Took {time() - start_time} seconds")


if __name__ == "__main__":
    main()
