import logging
from functools import wraps

logger = logging.getLogger(__name__)


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def log_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} {result.shape}")
        return result

    return wrapper


def _reader_generator(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def raw_newline_count(fname):
    """
    Counts number of lines in file
    """
    f = open(fname, "rb")
    f_gen = _reader_generator(f.raw.read)
    return sum(buf.count(b"\n") for buf in f_gen)


def start_pipeline(dataf):
    return dataf.copy()
