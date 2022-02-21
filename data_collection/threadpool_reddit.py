import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from time import time

from download_reddit import (
    get_dates,
    get_dates_from_file,
    download_data,
    setup_download_dir,
)

logger = logging.getLogger(__name__)


def main():
    dts_file = None
    subreddit = "PrayerRequests"
    object_type = "submission"
    start_date = None
    end_date = "2020-12-31"

    download_dir = setup_download_dir(subreddit, object_type)
    dts = []
    if dts_file is not None:
        dts = get_dates_from_file(dts_file)
    else:
        dts = get_dates(start_date=start_date, end_date=end_date)

    ts = time()
    with ThreadPoolExecutor() as executor:
        fn = partial(download_data, download_dir, subreddit, object_type)
        executor.map(fn, dts)

    logger.info(f"Took {time() - ts} seconds")
    logger.info(f"Finished downloading {object_type}")


if __name__ == "__main__":
    main()
