import os
import logging
from time import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import math

import numpy as np
import modin.pandas as pd
import re
import string
from nltk.tokenize import TweetTokenizer, word_tokenize
import emoji

from utils.utils import init_logger, log_step
from preprocess.preprocess import preprocess_text

logger = logging.getLogger(__name__)
init_logger()

punct_chars = list(
    (
        set(string.punctuation)
        | {"’", "‘", "–", "—", "~", "|", "“", "”", "…", "'", "`", "_", "“"}
    )
    - set(["#"])
)
punct_chars.sort()
punctuation = "".join(punct_chars)
replace = re.compile("[%s]" % re.escape(punctuation))

tokenizer = TweetTokenizer().tokenize


def get_filepaths(subreddits):
    filepaths = []
    for subreddit in subreddits:
        filepaths.extend(
            list(Path(f"../data/reddit/subreddit/{subreddit}/").glob("**/*.json"))
        )
    logger.info(f"Total no of files found: {len(filepaths)}")
    return filepaths


def read_file(fname):
    data = []
    with open(fname, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                dt = pd.to_datetime(rec["created_date"])
                text = rec["text"]
                if dt.year >= 2011 and dt.year < 2021:
                    if not (text == "[removed]" or text == "[deleted]"):
                        data.append(rec)
            except json.decoder.JSONDecodeError:
                pass
    return data


def read(filepaths):
    ts = time()

    logger.info(" ***** Reading files *****")
    logger.info(f" Num of files = {len(filepaths)}")
    with ProcessPoolExecutor() as executor:
        data = executor.map(read_file, filepaths)

    logger.info(f"Took {time() - ts}")
    logger.info(" ***** Finished reading files *****")

    dataf = pd.DataFrame(itertools.chain(*data))

    return dataf


@log_step
def remove_duplicates(dataf):
    return dataf[~dataf["id"].duplicated(keep="first")].reset_index(drop=True)


@log_step
def normalize_texts(dataf):
    """Normalize text."""

    def _normalize_text(text):
        text = text.replace("\n", "")
        text = re.sub("\s+", " ", text)
        text = re.sub(
            r"http\S*|\S*\.com\S*|\S*www\S*", "[URL]", text
        )  # replace urls with [URL] tag
        return text

    texts = dataf["text"].apply(_normalize_text)
    dataf["text"] = texts
    return dataf


def _get_tokenizer(tokenizer):
    if tokenizer == "tweet":
        return TweetTokenizer().tokenize
    else:
        return word_tokenize


@log_step
def add_token_len(dataf):
    token_lengths = dataf.text.apply(
        lambda x: len(tokenizer(x.replace("[URL]", "URL")))
    )
    dataf["token_len"] = token_lengths

    return dataf


@log_step
def filter_token_len(dataf, min=3, max=30):
    if "token_len" not in dataf:
        token_lengths = dataf["text"].apply(
            lambda x: len(tokenizer(x.replace("[URL]", "URL")))
        )
        dataf["token_len"] = token_lengths

    return dataf.loc[lambda d: (d["token_len"] >= min) & (d["token_len"] <= max)]


def CleanText(text):
    """Clean text."""
    if isinstance(text, float):
        return []
    text = text.lower()  # lower case
    text = text.replace("[removed]", " ")
    text = text.replace("[deleted]", " ")
    text = re.sub(r"http\S*|\S*\.com\S*|\S*www\S*", " ", text)  # remove urls
    text = re.sub(r"\s@\S+", " ", text)
    text = replace.sub(" ", text)  # substitute all other punctuation with whitespace
    text = re.sub(r"\s+", " ", text)  # replace all whitespace with a single space
    text = text.strip()
    words = text.split()
    return [w for w in words if len(w) > 2]


def text_has_user_mention(text):
    if re.search(r"\s@\S+", text):
        return True
    return False


def text_has_emoji(text):
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            return True
    return False


@log_step
def _rebalance(dataf):
    class_dist = dict(dataf["label"].value_counts())
    minority_class = min(class_dist, key=class_dist.get)
    num_minority = class_dist[minority_class]
    other_class_dist = {k: class_dist[k] for k in class_dist if k != minority_class}
    target_sample = dataf[dataf["label"] == minority_class]
    for cls in other_class_dist:
        num_unique_dates = dataf[dataf["label"] == cls].created_date.unique().shape[0]
        num_sample_per_date = math.ceil((num_minority / num_unique_dates))
        sample = (
            dataf[dataf["label"] == cls]
            ._to_pandas()
            .groupby("created_date")
            .sample(num_sample_per_date, random_state=42, replace=True)
        )
        sample = pd.DataFrame(sample)
        target_sample = target_sample.append(sample).reset_index(drop=True)
    return target_sample


def create_splits(dataf, dir_to_save_splits, columns=["text", "label"]):
    dataf = dataf[columns]
    os.makedirs(dir_to_save_splits, exist_ok=True)
    train, validate, test = np.split(
        dataf.sample(frac=1, random_state=42),
        [int(0.6 * len(dataf)), int(0.8 * len(dataf))],
    )
    train.to_json(
        os.path.join(dir_to_save_splits, "train.json"), orient="records", lines=True
    )
    validate.to_json(
        os.path.join(dir_to_save_splits, "valid.json"), orient="records", lines=True
    )
    test.to_json(
        os.path.join(dir_to_save_splits, "test.json"), orient="records", lines=True
    )


def write_sentences_by_subreddit(dataf, subreddit, output_dir, column="label"):
    dataf = dataf[dataf[column] == dataf]
    path = os.path.join(output_dir, f"{subreddit}_keys.txt")
    with open(path, "w") as f:
        for text in dataf["text"].tolist():
            f.write(text + "\n")


@log_step
def add_label_column(dataf, label):
    dataf["label"] = label
    return dataf


def add_cos_score(dataf, fpath):
    cos_scores = []
    with open(fpath, "r") as f:
        for line in f:
            idx, score = line.split(",")
            cos_scores.append([int(idx), float(score)])

    cos_scores = dataf.index.to_series().apply(lambda x: cos_scores[x][1])
    dataf["cos_score"] = cos_scores

    return dataf


@log_step
def filter_cos_score(dataf, frac):
    if frac > 1:
        top_k = frac
    else:
        top_k = math.ceil((frac * len(dataf)))
    dataf.sort_values(by=["cos_score"], ascending=False, inplace=True)
    return dataf.iloc[:top_k]


@log_step
def select_columns(dataf, columns=["text", "label"]):
    return dataf[columns]


@log_step
def preprocess_function(dataf):
    texts = dataf.text.apply(preprocess_text)
    dataf["text"] = texts
    return dataf.dropna(subset=["text"])
