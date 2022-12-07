# code adapted from https://github.com/google-research/google-research/blob/master/goemotions/extract_words.py

import argparse
import glob
import os

from collections import defaultdict, Counter
from itertools import chain
import string
import re
import math
import operator

import json
import modin.pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", default="./data/reddit", help="Directory containing full dataset."
)
parser.add_argument(
    "--output",
    default="religion_words.csv",
    help="Output csv file for the religion words.",
)
parser.add_argument(
    "--religion-file",
    default="./data/reddit/subreddit.txt",
    help="File contating list of religions.",
)

args = parser.parse_args()

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


def LogOdds(counts1, counts2, prior, zscore=True):
    """Calculates log odds ratio.
    Source: Dan Jurafsky.
    Args:
      counts1: dict of word counts for group 1
      counts2: dict of word counts for group 2
      prior: dict of prior word counts
      zscore: whether to z-score the log odds ratio
    Returns:
      delta: dict of delta values for each word.
    """

    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    n1 = sum(counts1.values())
    n2 = sum(counts2.values())

    nprior = sum(prior.values())
    for word in prior.keys():
        if prior[word] == 0:
            delta[word] = 0
            continue
        l1 = float(counts1[word] + prior[word]) / (
            (n1 + nprior) - (counts1[word] + prior[word])
        )
        l2 = float(counts2[word] + prior[word]) / (
            (n2 + nprior) - (counts2[word] + prior[word])
        )
        sigmasquared[word] = 1 / (float(counts1[word]) + float(prior[word])) + 1 / (
            float(counts2[word]) + float(prior[word])
        )
        sigma[word] = math.sqrt(sigmasquared[word])
        delta[word] = math.log(l1) - math.log(l2)
        if zscore:
            delta[word] /= sigma[word]
    return delta


def GetCounts(df):
    words = []
    for t in df["text"]:
        words.extend(t)
    return Counter(words)


def read_json_data(fname):
    with open(fname, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                yield rec
            except json.decoder.JSONDecodeError:
                pass


def main():
    print("Loading data...")
    religions = []
    with open(args.religion_file, "r") as f:
        religions = f.read().splitlines()

    filenames = []
    for religion in religions:
        filenames.extend(glob.glob(f"{args.data}/{religion}/*.csv"))

    # json_data = []
    # for fname in filenames:
    #     json_data.append(read_json_data(fname))

    dfs = []
    for fname in filenames:
        dfs.append(pd.read_csv(fname, encoding="utf-8", lineterminator="\n"))

    data = pd.concat(dfs)
    print(f"{data.shape[0]} Examples")

    print("Processing data.....")
    data["text"] = data["text"].apply(CleanText)

    data = data[data["text"].apply(lambda text: True if len(text) > 0 else False)]
    data = data[~data["religion"].isnull()]
    dicts = []
    for religion in religions:
        print(religion)
        contains = data["religion"].str.contains(religion)
        religion_words = GetCounts(data[contains])
        other_words = GetCounts(data[~contains])
        prior = Counter()
        prior.update(dict(religion_words))
        prior.update(dict(other_words))
        religion_words_total = sum(religion_words.values())
        delta = LogOdds(religion_words, other_words, prior, True)
        c = 0
        for k, v in sorted(delta.items(), key=operator.itemgetter(1), reverse=True):
            if v < 3:
                continue
            dicts.append(
                {
                    "religion": religion,
                    "word": k,
                    "odds": "%.2f" % v,
                    "freq": "%.3f" % (religion_words[k] / religion_words_total),
                }
            )
            c += 1
            if c < 11:
                print("%s (%.2f)" % (k, v))
            print("--------")

    # if not os.path.isdir(args.output):
    #     os.makedirs(args.output)

    religion_words_df = pd.DataFrame(dicts)
    religion_words_df.to_csv(args.output, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
