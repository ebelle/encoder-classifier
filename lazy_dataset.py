import torch
from torch.utils.data import Dataset
import csv
import os
import linecache
import subprocess


class LazyDataset(Dataset):
    def __init__(self, data_path, filename, source_vocab, target_vocab, task):
        self.data_path = data_path
        self.source_vocab = source_vocab
        self.text_init = self.source_vocab.init_token
        self.text_eos = self.source_vocab.eos_token
        self.target_vocab = target_vocab
        self.target_init = self.target_vocab.init_token
        self.target_eos = self.target_vocab.eos_token
        self._filepath = os.path.join(data_path, filename)
        # get total file length
        self._total_data = int(
            subprocess.check_output("wc -l " + self._filepath, shell=True).split()[0]
        )
        self.task = task

    def __len__(self):
        "Denotes the total number of samples"
        return self._total_data

    def tokens_to_idx(self, text, target):
        # TODO: add arguments to make init & eos optional
        # add init and eos tokens
        if self.task == "translation":
            text = [self.text_init] + text + [self.text_eos]
            target = [self.target_init] + target + [self.target_eos]
        # tokens to indices
        text = [self.source_vocab.vocab.stoi[t] for t in text]
        target = [self.target_vocab.vocab.stoi[t] for t in target]

        return text, target

    def __getitem__(self, index):
        "Generates one sample of data"
        # skip header
        if index != 0:
            line = linecache.getline(self._filepath, index + 1)
            text, target = line.split("\t")
            # string to list, tokenizing on white space
            text, target = text.split(), target.split()
            text, target = self.tokens_to_idx(text, target)
            text_lens = len(text)
            return text, target, text_lens
