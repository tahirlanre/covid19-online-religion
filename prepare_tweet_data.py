import os
import time
from pathlib import Path
from itertools import chain
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd


from utils.utils import init_logger
from preprocess.preprocess import get_tweet_data, normalize_tweet_text

logger = logging.getLogger(__name__)

init_logger()


def clean_texts(dataf):
    texts = dataf["text"].apply(normalize_tweet_text)
    df["text"] = texts
    return dataf.dropna(subset=["text"])


for year in ["2020"]:
    for month in ["07", "08", "09"]:
        data1 = Path(f"data/twitter/UK/{year}/{month}/raw/")
        output_dir = Path(f"data/twitter/UK/clean/{year}/{month}")

        filepaths = []
        filepaths.append(data1.glob("*.jsonl"))

        filepaths = list(chain(*filepaths))
        logger.info(f"***** Total no of filepaths: {len(filepaths)} *****")

        columns_to_get = ["id", "text", "author_id", "created_at"]
        fn = partial(get_tweet_data, columns=columns_to_get)

        logger.info(f"***** Processing tweet data from {month}-{year} *****")
        ts = time.time()
        with ProcessPoolExecutor() as executor:
            data = executor.map(fn, filepaths)

        df = pd.DataFrame(list(chain(*data)), columns=columns_to_get)

        clean_df = df.pipe(clean_texts)
        logger.info(f"Finished processing all tweets. Took {time.time() - ts} s")

        out_file = os.path.join(output_dir, "data.json")
        clean_df.to_json(out_file, orient="records", lines=True)
