import os
import random
import logging
from pathlib import Path
import shutil

import torch
import numpy as np

from sklearn.metrics import f1_score, accuracy_score


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    results = dict()

    results["accuracy"] = accuracy_score(labels, preds)
    results["f1"] = f1_score(labels, preds, average="micro")

    return results


def save_checkpoint(model_to_save, tokenizer, global_step, output_dir):
    # delete older checkpoint(s)
    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"checkpoint-*")]

    for checkpoint in glob_checkpoints:
        # logger.info(f"Deleting older checkpoint {checkpoint}")
        shutil.rmtree(checkpoint)

    # Save model checkpoint
    ckpt_output_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_output_dir, exist_ok=True)

    # logger.info(f"Saving model checkpoint to {ckpt_output_dir}")

    model_to_save.save_pretrained(ckpt_output_dir)
    tokenizer.save_pretrained(ckpt_output_dir)
