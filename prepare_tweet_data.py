import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
import logging

import modin.pandas as pd
import ray

from preprocess.preprocess import get_all_tweet_texts
from utils.utils import init_logger

ray.init()

init_logger()
logger = logging.getLogger(__name__)


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
ts = time.time()
with ProcessPoolExecutor() as executor:
    texts = executor.map(get_all_tweet_texts, filepaths)
logger.info(f"Took {time.time() - ts} seconds")

df = pd.DataFrame(list(chain(*texts)), columns=["text"]).dropna(subset=["text"])

# drop duplicates
df = df[~df["text"].duplicated(keep="first")].reset_index(drop=True)
