import re
import json

from emoji import demojize
from nltk.tokenize import TweetTokenizer

TWITTER_USER_RE = re.compile(r"""(?:@\w+)""", re.UNICODE)
REDDIT_USER_RE = re.compile(r"(?:\/?u\/\w+)", flags=re.UNICODE)
URL_RE = re.compile(
    r"""((https?:\/\/|www)|\w+\.(\w{2-3}))([\w\!#$&-;=\?\-\[\]~]|%[0-9a-fA-F]{2})+""",
    re.UNICODE,
)
mention_regex = re.compile(r"\s@\S+", re.UNICODE)
hashtag_regex = re.compile(r"\s#S+", re.UNICODE)

tokenizer = TweetTokenizer()


def normalize_tweet_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(TWITTER_USER_RE, "<user>", text)
    text = re.sub(URL_RE, "<url>", text)
    text = re.sub("\n", "", text)
    text = re.sub(r"\s+", " ", text)  # remove double or more space
    text = demojize(text)
    return text


def normalize_reddit_text(text):
    text = text.lower()
    text = text.replace("[deleted]", " ").replace("[removed]", " ")
    text = re.sub(REDDIT_USER_RE, "<user>", text)
    text = re.sub(URL_RE, "<url>", text)
    text = re.sub("\n", " ", text)
    text = re.sub(r"\s+", " ", text)  # remove double or more space
    text = demojize(text)
    return text


def has_mention(text):
    if mention_regex.search(text):
        return True
    return False


def has_hashtag(text):
    if hashtag_regex.search(text):
        return True
    return False


def get_tweet_data(filepath, columns=["id", "trxt"]):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            t = tuple()
            tweet_obj = json.loads(line)
            for col in columns:
                if col in tweet_obj:
                    t += (tweet_obj[col],)
            data.append(t)
    return data
