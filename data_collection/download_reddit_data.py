import os
import argparse

import requests
from datetime import datetime
import traceback
import time
import json
import sys

# subreddit = "Islam"  # put the subreddit you want to download in the quotes

url = "https://api.pushshift.io/reddit/{}/search?limit=1000&sort=desc&{}&before="


def downloadFromUrl(subreddit, object_type, start_time=None):
    filter_string = f"subreddit={subreddit}"
    output_dir = f"./data/reddit/{subreddit}/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    fname = f"{'post' if object_type == 'submission' else object_type}.json"
    output_file = os.path.join(output_dir, fname)
    print(f"Saving {object_type}s to {output_file}")
    count = 0
    if start_time is None:
        start_time = datetime.utcnow()
    previous_epoch = int(start_time.timestamp())

    while True:
        new_url = url.format(object_type, filter_string) + str(previous_epoch)
        json_text = requests.get(
            new_url, headers={"User-Agent": "Post downloader by Tahir"}
        )
        time.sleep(
            1
        )  # pushshift has a rate limit, if we send requests too fast it will start returning error messages
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
                    with open(output_file, "a+") as filehandle:
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
                        with open(output_file, "a+") as filehandle:
                            filehandle.write(f"{json.dumps(post)}\n")
                    except Exception as err:
                        print(f"Couldn't print post: {object['url']}")
                        print(traceback.format_exc())

        print(
            "Saved {} {}s through {}".format(
                count,
                object_type,
                datetime.fromtimestamp(previous_epoch).strftime("%Y-%m-%d"),
            )
        )

    print(f"Saved {count} {object_type}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reddit downloader")
    parser.add_argument(
        "--subreddit", help="the subreddit you want to download", required=True
    )
    parser.add_argument(
        "--start_time",
        type=datetime.fromisoformat,
        help="time (in YYYY-MM-D format) to download from",
    )
    parser.add_argument(
        "--post", help="download submissions from subreddit", action="store_true"
    )
    parser.add_argument(
        "--comment", help="download comments from subbredit", action="store_true"
    )
    args = parser.parse_args()

    subreddit = args.subreddit
    start_time = args.start_time

    if args.post == False and args.comment == False:
        raise ValueError(
            "Need to select either submissions or comments to download from subredit"
        )

    if args.post:
        print(f"Downloading posts from {subreddit} subreddit")
        downloadFromUrl(subreddit, "submission", start_time=start_time)
    if args.comment:
        print(f"Downloading comments from {subreddit} subreddit")
        downloadFromUrl(subreddit, "comment", start_time)
