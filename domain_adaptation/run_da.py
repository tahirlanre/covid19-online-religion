import os
import argparse
import logging
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import shutil
from pathlib import Path

import numpy as np
import math
from sklearn.metrics import classification_report
import datasets
from datasets import load_dataset
from dataset import TextDataset, DoubleSubsetRandomSampler
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    RandomSampler,
)

import transformers
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoConfig,
    default_data_collator,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)

import torch
from model import DoubleHeadBert

import wandb

from utils.train import compute_metrics

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


class DoubleCollator(object):
    def __init__(self, source_collate_fn, target_collate_fn):
        self.source_collate_fn = source_collate_fn
        self.target_collate_fn = target_collate_fn

    def __call__(self, batch):
        domains = [b["domain"] for b in batch]
        if domains[0] == 0:
            return self.source_collate_fn(batch)
        else:
            return self.target_collate_fn(batch)


def dataloaders_from_datasets(
    s_train_dataset, s_val_dataset, t_train_dataset, batch_size, circle, collate_fn
):
    train_dataset = ConcatDataset([s_train_dataset, t_train_dataset])
    s_dataset_size = len(s_train_dataset)
    s_train_indices = list(range(len(s_train_dataset)))

    t_train_indices = list(range(len(t_train_dataset)))

    train_sampler = DoubleSubsetRandomSampler(
        s_train_indices,
        t_train_indices,
        s_dataset_size,
        batch_size,
        batch_size * circle,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        s_val_dataset,
        batch_size=batch_size,
        # sampler=RandomSampler,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def get_mixed_loss(dict_loss):
    n = dict_loss[0]["count"]
    m = dict_loss[1]["count"]
    weight_factor = n / (n + m)
    source_loss = dict_loss[0]["loss"]
    target_loss = dict_loss[1]["loss"]
    return (weight_factor * source_loss) + ((1 - weight_factor) * target_loss)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_train_file",
        type=str,
        default=None,
        help="A csv or a json file contatining the training source data.",
    )
    parser.add_argument(
        "--source_validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation sourcr data.",
    )
    parser.add_argument(
        "--target_train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training target data.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Number of updates steps before two checkpoint saves",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Number of update steps between two logs",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use gpu",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, do_lower_case=True
    )

    data_files = {}
    data_files["source_train"] = args.source_train_file
    data_files["source_validation"] = args.source_validation_file
    data_files["target_train"] = args.target_train_file
    extension = (args.source_train_file).split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    label_list = raw_datasets["source_train"].unique("label")
    label_list.sort()
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    model = DoubleHeadBert.from_pretrained(args.model_name_or_path, config=config)
    model.resize_token_embeddings(len(tokenizer))

    device = "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu"

    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}
    model.to(device)

    wandb.login()
    wandb.init(project="religion-da", config=model.config)

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        texts = (examples["text"],)
        result = tokenizer(
            *texts, padding=padding, max_length=args.max_seq_length, truncation=True
        )
        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id.get(l, -1) for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]

        if result["labels"][0] == -1:
            result["domain"] = [1 for l in examples["label"]]
        else:
            result["domain"] = [0 for l in examples["label"]]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["source_train"].column_names,
        desc="Running tokenizer on dataset",
    )

    target_train_dataset = processed_datasets["target_train"]
    source_train_dataset = processed_datasets["source_train"]
    source_eval_dataset = processed_datasets["source_validation"]

    source_collate_fn = default_data_collator
    target_collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer)
    collate_fn = DoubleCollator(source_collate_fn, target_collate_fn)

    circle = math.floor(
        (
            len(target_train_dataset)
            / (len(source_train_dataset) / args.per_device_train_batch_size)
        )
        / args.per_device_train_batch_size
    )

    train_dataloader, eval_dataloader = dataloaders_from_datasets(
        source_train_dataset,
        source_eval_dataset,
        target_train_dataset,
        batch_size=args.per_device_train_batch_size,
        circle=circle,
        collate_fn=collate_fn,
    )

    args.gradient_accumulation_steps = circle + 1

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    total_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num source examples = {len(source_train_dataset)}")
    logger.info(f"  Num target examples = {len(target_train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    progress_bar = tqdm(range(max_train_steps))
    global_step = 0
    tr_loss = 0.0
    best_val_loss = float("inf")

    run_name = (
        wandb.run.name if wandb.run else datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    )
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, run_name)

    train_loss = 0.0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: batch[k].to(device) for k in batch}
            loss, output = model(**batch)
            loss = loss / args.gradient_accumulation_steps

            loss.backward()
            train_loss += loss.item()
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    wandb.log({"train_loss": train_loss / global_step})

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.output_dir is not None:
                        # delete older checkpoint(s)
                        glob_checkpoints = [
                            str(x) for x in Path(output_dir).glob(f"checkpoint-*")
                        ]

                        for checkpoint in glob_checkpoints:
                            logger.info(f"Deleting older checkpoint {checkpoint}")
                            shutil.rmtree(checkpoint)

                        # Save model checkpoint
                        ckpt_output_dir = os.path.join(
                            output_dir, f"checkpoint-{global_step}"
                        )
                        os.makedirs(ckpt_output_dir, exist_ok=True)

                        logger.info(f"Saving model checkpoint to {ckpt_output_dir}")

                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        model_to_save.save_pretrained(ckpt_output_dir)
                        tokenizer.save_pretrained(ckpt_output_dir)

        eval_loss = 0.0
        y_pred = None
        y_true = None

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                batch = {k: batch[k].to(device) for k in batch}
                loss, output = model(**batch)
                eval_loss += loss.item()

                logits = output
                if y_pred is None:
                    y_pred = logits.argmax(dim=-1).detach().cpu().numpy()
                    y_true = batch["labels"].detach().cpu().numpy()
                else:
                    y_pred = np.append(
                        y_pred, logits.argmax(dim=-1).detach().cpu().numpy(), axis=0
                    )
                    y_true = np.append(y_true, batch["labels"].detach().cpu().numpy())

        eval_loss = eval_loss / len(eval_dataloader)
        logger.info(f"Eval loss: {eval_loss}")
        wandb.log({"eval_loss": eval_loss})

        eval_metric = classification_report(y_true, y_pred)

        if args.output_dir is not None:
            eval_output_dir = os.path.join(output_dir, "eval")
            os.makedirs(eval_output_dir, exist_ok=True)

            output_eval_file = os.path.join(
                eval_output_dir, f"eval_{epoch+1}.txt" if global_step else "eval.txt"
            )
            with open(output_eval_file, "w") as f_w:
                logger.info(
                    f"*****  Evaluation results on eval dataset - Epoch: {epoch+1} *****"
                )
                f_w.write(eval_metric + "\n")
                f_w.write(f" eval_loss = {eval_loss}")

        if eval_loss < best_val_loss:
            if args.output_dir is not None:
                logger.info(f"Saving best model to {output_dir}")
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
