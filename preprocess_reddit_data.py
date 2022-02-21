import logging
from pathlib import Path
from time import time
from concurrent.futures import ProcessPoolExecutor

import json
from itertools import chain
import modin.pandas as pd
import ray

import re

ray.init()


logging.basicConfig(
    level=logging.info, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def read_json_data(path):
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
                    #  discard posts that have been removed or deleted
                    if not (text == "[removed]") or (text == "[deleted]"):
                        data.append(rec)
            except json.decoder.JSONDecodeError:
                logger.error("Error reading file")
    return data


def normalize_text(text):
    """Normalize text."""
    text = text.replace("\n", "")
    text = re.sub("\s+", " ", text)
    text = re.sub(
        r"http\S*|\S*\.com\S*|\S*www\S*", "URL", text
    )  # replace urls with URL tag
    # text = re.sub(r"\s@\S+", "USER", text) # user ment
    return text


def main():
    # get list of subreddits
    subreddits = []
    with open("data/reddit/subreddit.txt", "r") as f:
        subreddits = f.read().splitlines()

    #  get file paths to subreddits posts and comments
    json_files = []
    for subreddit in subreddits:
        json_files.extend(list(Path(f"data/reddit/{subreddit}/").glob("*.json")))

    # get data from all subreddits
    logger.info("Reading json data from file .......")
    start_time = time()
    with ProcessPoolExecutor() as executor:
        data = executor.map(read_json_data, json_files)
    logger.info(f"Took {time() - start_time} seconds")
    data = list(chain(*data))

    df = pd.DataFrame(data)

    # drop duplicates posts
    df = df[df["id"].duplicated(keep="first")].reset_index(drop=True)


if __name__ == "__main__":
    main()
