import torch
from torch.utils.data import Dataset
import csv
import os
import linecache


class LazyDataset(Dataset):
    def __init__(self, filepath, source_vocab, target_vocab, task):
        self.source_vocab = source_vocab
        self.text_init = self.source_vocab.init_token
        self.text_eos = self.source_vocab.eos_token
        self.target_vocab = target_vocab
        self.target_init = self.target_vocab.init_token
        self.target_eos = self.target_vocab.eos_token
        self.filepath = filepath
        self.task = task

    def tokens_to_idx(self, text, target):
        # TODO: add arguments to make init & eos optional
        # add init and eos tokens
        if self.task == "evaluation":
            text = [self.text_init] + text + [self.text_eos]
            # tokens to indices
            text = [self.source_vocab.vocab.stoi[t] for t in text]
        elif self.task == "translation":
            text = [self.text_init] + text + [self.text_eos]
            target = [self.target_init] + target + [self.target_eos]
            # tokens to indices
            text = [self.source_vocab.vocab.stoi[t] for t in text]
            target = [self.target_vocab.vocab.stoi[t] for t in target]
        elif self.task == "tagging":
            # tokens to indices
            text = [self.source_vocab.vocab.stoi[t] for t in text]
            target = [self.target_vocab.vocab.stoi[t] for t in target]
        else:
            print(f"{self.task} is not an option")
            quit()

        return text, target

    def __getitem__(self, idx):
        "Generates one sample of data"
        # normally you need +1 since linecache indexes from 1
        # here, we skip the header by adding +2 instead of +1
        line = linecache.getline(self.filepath, idx + 2)
        text, target = line.split("\t")

        # string to list, tokenizing on white space
        text = text.split()
        target = target.split()
        text, target = self.tokens_to_idx(text, target)
        text_lens = len(text)

        if self.task == "evaluation":
            target = idx
        return text, target, text_lens
