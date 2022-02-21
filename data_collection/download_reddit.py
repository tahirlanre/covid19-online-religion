import logging
import os
from pathlib import Path

import requests
from datetime import datetime, timedelta
import traceback
import time
import json
import pandas as pd

url = "https://api.pushshift.io/reddit/{}/search?limit=1000&sort=desc&{}&after{}&before={}"

logger = logging.getLogger(__name__)


def get_dates(start_date=None, end_date=None):
    if start_date is not None:
        start_date = pd.to_datetime(start_date).date()
    else:
        start_date = datetime(2011, 1, 1).date()

    if end_date is not None:
        end_date = pd.to_datetime(end_date).date()
    else:
        end_date = datetime.now().date()

    dts = pd.date_range(start_date, end_date, freq="d").strftime("%Y-%m-%d").tolist()

    return dts


def get_dates_from_file(filename):
    dts = []
    with open(filename, "r") as f:
        dts = f.read().split()
    return dts


def download_data(directory, subreddit, object_type, dt):
    filter_string = f"subreddit={subreddit}"
    download_path = os.path.join(directory, f"{dt}.json")
    count = 0

    start_time = datetime.fromisoformat(dt)
    end_time = start_time + timedelta(days=1)

    start_time = int(start_time.timestamp())
    end_time = int(end_time.timestamp())

    previous_epoch = end_time

    while True:
        new_url = url.format(object_type, filter_string, start_time - 1, previous_epoch)
        try:
            json_text = requests.get(
                new_url, headers={"User-Agent": "Post downloader by Tahir"}
            )
        except requests.exceptions.ChunkedEncodingError as e:
            logger.info("sleeping for 60 secs .........")
            time.sleep(60)
            continue

        # pushshift has a rate limit, if we send requests too fast it will start returning error messages
        # logger.info("sleeping for 1 sec .........")
        time.sleep(1)

        try:
            json_data = json_text.json()
        except json.decoder.JSONDecodeError:
            time.sleep(1)
            continue

        if "data" not in json_data:
            break
        objects = json_data["data"]
        if len(objects) == 0:
            break
        for object in objects:
            previous_epoch = object["created_utc"] - 1
            count += 1
            if object_type == "comment":
                try:
                    comment = {}
                    comment["id"] = object["id"]
                    comment["author"] = object["author"]
                    comment["created_date"] = datetime.fromtimestamp(
                        object["created_utc"]
                    ).strftime("%Y-%m-%d")
                    comment["text"] = (
                        object["body"]
                        .encode(encoding="ascii", errors="ignore")
                        .decode()
                    )
                    comment["subreddit"] = object["subreddit"]
                    comment["url"] = (
                        f"https://www.reddit.com{object['permalink']}"
                        if "permalink" in object
                        else ""
                    )
                    comment["author_flair_text"] = object["author_flair_text"]
                    comment["score"] = object["score"]
                    with open(download_path, "a+") as filehandle:
                        filehandle.write(f"{json.dumps(comment)}\n")
                except Exception as err:
                    if "permalink" in object:
                        print(
                            f"Couldn't print comment: https://www.reddit.com{object['permalink']}"
                        )
                    else:
                        print("Couldn't get comment")
                    print(traceback.format_exc())
            elif object_type == "submission":
                if object["is_self"]:
                    if "selftext" not in object:
                        continue
                    try:
                        post = {}
                        post["id"] = object["id"]
                        post["author"] = object["author"]
                        post["created_date"] = datetime.fromtimestamp(
                            object["created_utc"]
                        ).strftime("%Y-%m-%d")
                        post["text"] = (
                            object["selftext"]
                            .encode(encoding="ascii", errors="ignore")
                            .decode()
                        )
                        post["subreddit"] = object["subreddit"]
                        post["url"] = object["url"]
                        post["author_flair_text"] = object["author_flair_text"]
                        post["score"] = object["score"]
                        with open(download_path, "a+") as filehandle:
                            filehandle.write(f"{json.dumps(post)}\n")
                    except Exception as err:
                        print(f"Couldn't print post: {object['url']}")
                        print(traceback.format_exc())

        if previous_epoch < start_time:
            break

    logger.info(
        "Saved {} {}s through {}".format(
            count,
            object_type,
            datetime.fromtimestamp(start_time).strftime("%Y-%m-%d"),
        )
    )

    # logger.info(f"")


def setup_download_dir(subreddit, object_type):
    download_dir = Path(
        f"../data/reddit/{subreddit}/{'post' if object_type == 'submission' else object_type}"
    )
    if not download_dir.exists():
        download_dir.mkdir(parents=True, exist_ok=True)
    return download_dir
