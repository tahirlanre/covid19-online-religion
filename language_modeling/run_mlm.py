import argparse
import os
import logging
from itertools import chain
import random
import math
from datetime import datetime

import transformers
from transformers import (
    CONFIG_MAPPING,
    SchedulerType,
    set_seed,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AdamW,
    get_scheduler,
)

import datasets
from datasets import load_dataset

from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from utils import save_checkpoint

import wandb

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a transformers model on Masked Language Modeling task"
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
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
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
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
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
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
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
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Number of updates steps before two checkpoint saves",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this "
        "value if set.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of evaluation examples to this "
        "value if set.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.train_file is not None:
        extension = args.train_file.split(".")[-1]
        if extension not in ["csv", "json", "txt"]:
            raise ValueError("`train_file` should be a csv, json or txt file.")
    if args.validation_file is not None:
        extension = args.validation_file.split(".")[-1]
        if extension not in ["csv", "json", "txt"]:
            raise ValueError("`validation_file` should be a csv, json or txt file.")

    return args


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.setLevel(logging.INFO)

    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    raw_datasets = load_dataset(extension, data_files=data_files)

    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    # if "validation" not in raw_datasets.keys():
    #     raw_datasets["validation"] = load_dataset(
    #         extension,
    #         data_files=data_files,
    #         split=f"train[:{args.validation_split_percentage}%]"
    #     )
    #     raw_datasets["train"] = load_dataset(
    #         extension,
    #         data_files=data_files,
    #         split=f"train[{args.validation_split_percentage}%:]"
    #     )

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("Instantiating a new config instance from scratch")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        special_tokens_dict = {'additional_special_tokens': ['[URL]']}
        tokenizer.add_special_tokens(special_tokens_dict)
    else:
        logger.info("Training new tokenizer from scratch")

        special_tokens = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[USER]",
            "[URL]",
        ]

        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.Sequence(
            [NFD(), Lowercase(), StripAccents()]
        )
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )
        trainer = WordPieceTrainer(
            vocab_size=30522,
            special_tokens=special_tokens,
        )
        tokenizer.train([args.train_file], trainer)
        tokenizer_path = os.path.join(os.path.dirname(args.train_file), "uktweetsbert")
        os.mkdir(tokenizer_path, exist_ok=True)
        tokenizer.save(tokenizer_path)

        tokenizer = Tokenizer.from_file(tokenizer_path)

    if args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    wandb.login()
    wandb.init(project="twitter-lm", config=model.config)

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 128:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 128 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 128
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    if args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if args.pad_to_max_length else False

        def tokenize_function(examples):
            examples[text_column_name] = [
                line
                for line in examples[text_column_name]
                if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            desc="Running tokenizer on dataset line_by_line",
        )
    else:

        def tokenize_function(examples):
            return tokenizer(
                examples[text_column_name], return_special_tokens_mask=True
            )

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on every text in dataset",
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [
                    t[i : i + max_seq_length]
                    for i in range(0, total_length, max_seq_length)
                ]
                for k, t in concatenated_examples.items()
            }
            return result

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )

    train_dataset = tokenized_datasets["train"]
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))

    eval_dataset = tokenized_datasets["validation"]
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=args.mlm_probability
    )

    # DataLoaders creation:
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

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
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
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    global_step = 0

    run_name = wandb.run.name if wandb.run else datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, run_name)

    train_loss = 0.0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            train_loss += loss
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
                    model_to_save = model.module if hasattr(model, "module") else model
                    save_checkpoint(model_to_save, tokenizer, global_step, output_dir)
                
                # log interval loss
                train_loss = train_loss / args.save_steps
                wandb.log({"train_loss": train_loss})
                train_loss = 0.
            
            # if completed_steps >= args.max_train_steps:
            #     break

        model.eval()
        eval_loss = 0.
        for step, batch in enumerate(eval_dataloader):
            batch.to(device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            eval_loss += loss.item()
        eval_loss = eval_loss / len(eval_dataloader)
        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        wandb.log({"perplexity": perplexity})
        logger.info(f"epoch {epoch}: perplexity: {perplexity}")

    if args.output_dir is not None:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
