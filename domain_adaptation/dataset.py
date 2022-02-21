import os

import numpy as np
import json

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

import torch
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(
        self,
        ds,
        tokenizer,
        split="train",
        dl=0,
        max_seq_length=50,
        labeled=True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.labels = []
        self.texts = []
        self.domains = []
        self.domain = dl
        file = os.path.join(ds, f"{split}.json")
        with open(file) as f:
            for line in f:
                example = json.loads(line)
                label = example["label"] if labeled else -1
                self.labels.append(label)
                text = self.tokenizer.encode(
                    example["text"],
                    add_special_tokens=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                )
                self.texts.append(text)
                self.domains.append(self.domain)
        if labeled:
            label_list = list(set(self.labels))
            label_to_id = {v: i for i, v in enumerate(label_list)}
            self.labels = [label_to_id[label] for label in self.labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        domain = self.domains[idx]
        return text, label, domain


class DoubleSubsetRandomSampler(Sampler):
    def __init__(
        self, indices_source, indices_target, s_dataset_size, num_source, num_target
    ):
        self.indices_source = indices_source
        self.indices_target = indices_target
        self.s_dataset_size = s_dataset_size
        self.num_source = num_source
        self.num_target = num_target

    def __iter__(self):
        perm = torch.randperm(len(self.indices_source))
        tarperm = torch.randperm(len(self.indices_target))
        T = 0
        t = 0
        for i, s in enumerate(perm, 1):
            yield self.indices_source[s]
            if i % self.num_source == 0:
                for j in range(self.num_target):
                    t = T + j
                    yield self.s_dataset_size + self.indices_target[tarperm[t]]
                T = t + 1

    def __len__(self):
        full = int(
            np.floor(
                (len(self.indices_source) + len(self.indices_target)) / self.num_source
            )
        )
        last = len(self.indices_source) % self.num_source
        return int(full * self.num_source + last)
