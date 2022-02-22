from tqdm import tqdm
import logging

from datasets import load_dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
)

import torch
from torch.utils.data import DataLoader
import numpy as np

from train_utils import init_logger

logger = logging.getLogger(__name__)


def load_data(file_path):
    data_files = {}
    data_files["target"] = file_path
    ext = file_path.split(".")[-1]
    raw_datasets = load_dataset(ext, data_files=data_files)

    return raw_datasets


def preprocess_function(examples, tokenizer, padding="max_length", max_length=50):
    texts = (examples["text"],)
    result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)
    return result


def load_model(model_path):
    return AutoModelForSequenceClassification.from_pretrained(model_path)


def run_model(model, dataloader, device):
    progress_bar = tqdm(len(dataloader))
    y_pred = None
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: batch[k].to(device) for k in batch}
            outputs = model(**batch)
            if y_pred is None:
                y_pred = outputs.logits.detach().cpu().numpy()
            else:
                y_pred = np.append(
                    y_pred, outputs.logits.detach().cpu().numpy(), axis=0
                )
            progress_bar.update(1)

    return y_pred


def main():
    init_logger()

    month = "07"
    year = "2020"
    model_path = "saved_output/icy-leaf-4"
    data_file = "../data/twitter/UK/2020/sample/dev.json"
    out_file = f"./{month}_{year}_predictions.txt"

    raw_datasets = load_data(data_file)
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    processed_datasets = raw_datasets.map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=raw_datasets["target"].column_names,
        desc="Running tokenizer on dataset",
    )

    data_collator = default_data_collator
    batch_size = 64
    dataloader = DataLoader(
        processed_datasets["target"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

    model = load_model(model_path)
    model.resize_token_embeddings(len(tokenizer))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    logger.info("***** Running predict *****")
    logger.info(f"   Num examples = {len(processed_datasets['target'])}")
    y_pred = run_model(model, dataloader, device)

    logger.info("Writing predictions to file")
    with open(out_file, "w") as f_w:
        f_w.write("index\tprediction\n")
        for idx, item in enumerate(y_pred):
            f_w.write(f"{idx}\t{item}\n")


if __name__ == "__main__":
    main()
