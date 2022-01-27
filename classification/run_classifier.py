import argparse
import logging
from datetime import datetime
import math
import os
import random
from pathlib import Path
import shutil

import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

import numpy as np

import torch
import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    SchedulerType,
    DataCollatorWithPadding,
    default_data_collator,
    get_scheduler,
)

from classification.train_utils import (
    init_logger,
    set_seed,
    compute_metrics
)

import wandb

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformer model on text classification task"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file contatining the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="A csv or a json file containing the test data.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
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
        default=5e-5,
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
        "--output_dir", type=str, default=None, help="Where to store the final model."
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
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    init_logger()
    set_seed(args.seed)

    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file
    extension = (
        args.train_file if args.train_file is not None else args.valid_file
    ).split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    label_list = raw_datasets["train"].unique("label")
    label_list.sort()
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    special_tokens_dict = {'additional_special_tokens': ['[URL]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}
    model.to(device)

    wandb.login()
    wandb.init(project="reddit-religion", config=model.config)

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        texts = (examples["text"],)
        result = tokenizer(
            *texts, padding=padding, max_length=args.max_length, truncation=True
        )

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

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
    logger.info(f"  Num examples = {len(train_dataset)}")
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
    best_val_loss = float("inf")

    run_name = wandb.run.name if wandb.run else datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, run_name)

    train_loss = 0.
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            train_loss += loss.item()
            loss.backward()
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_step += 1

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
                    ckpt_output_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_output_dir, exist_ok=True)

                    logger.info(f"Saving model checkpoint to {ckpt_output_dir}")

                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(ckpt_output_dir)
                    tokenizer.save_pretrained(ckpt_output_dir)
                
                # log interval loss
                cur_loss = train_loss / args.save_steps
                wandb.log({"train_loss": cur_loss})
                train_loss = 0.
                
            # if global_step >= args.max_train_steps:
            #     break

        model.eval()
        eval_loss = 0.0
        y_pred = None
        y_true = None

        for step, batch in enumerate(eval_dataloader):
            batch.to(device)
            outputs = model(**batch)
            eval_loss += outputs.loss.item()
            if y_pred is None:
                y_pred = outputs.logits.argmax(dim=-1).detach().cpu().numpy()
                y_true = batch["labels"].detach().cpu().numpy()
            else:
                y_pred = np.append(y_pred, outputs.logits.argmax(dim=-1).detach().cpu().numpy(), axis=0)
                y_true = np.append(y_true, batch["labels"].detach().cpu().numpy())

        eval_metric = compute_metrics(y_true, y_pred)
        eval_loss = eval_loss / len(eval_dataloader)
        wandb.log({"eval_loss": eval_loss})

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
                for key in sorted(eval_metric.keys()):
                    logger.info(f" {key} = {str(eval_metric[key])}")
                    f_w.write(f" {key} = {str(eval_metric[key])}\n")
                logger.info(f" eval_loss = {eval_loss}")
                f_w.write(f" eval_loss = {eval_loss}\n")

        if eval_loss < best_val_loss:
            if args.output_dir is not None:
                logger.info(f"Saving best model to {output_dir}")
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

    checkpoint = os.path.join(output_dir)
    logger.info(f"Loading best model from {checkpoint}")
    best_model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    best_model.to(device)

    y_pred = None
    y_true = None
    for step, batch in enumerate(test_dataloader):
        batch.to(device)
        outputs = best_model(**batch)
        if y_pred is None:
                y_pred = outputs.logits.argmax(dim=-1).detach().cpu().numpy()
                y_true = batch["labels"].detach().cpu().numpy()
        else:
            y_pred = np.append(y_pred, outputs.logits.argmax(dim=-1).detach().cpu().numpy(), axis=0)
            y_true = np.append(y_true, batch["labels"].detach().cpu().numpy())

    test_metric = classification_report(y_true, y_pred, target_names=label_list)

    output_test_file = os.path.join(output_dir, "test_results.txt")
    output_prediction_file = os.path.join(output_dir, "test_predictions.txt")
    with open(output_test_file, "w") as f_w:
        logger.info(f"*****  Evaluation results on test dataset *****")
        f_w.write(test_metric)
        logger.info(f"{test_metric}")
        # for key in sorted(test_metric.keys()):
        #     logger.info(f" {key} = {str(test_metric[key])}")
        #     f_w.write(f" {key} = {str(test_metric[key])}\n")
    with open(output_prediction_file, "w") as f_w:
        f_w.write("index\tprediction\n")
        for index, item in enumerate(y_pred):
            item = label_list[item]
            f_w.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()
