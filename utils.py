import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


def random_init_weights(model):
    """randomly initialize weights"""
    for name, p in model.named_parameters():
        if "weight" in name:
            if p.requires_grad:
                nn.init.normal_(p.data, mean=0, std=0.01)
        else:
            nn.init.constant_(p.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prep_batch(batch, device):
    """pad source and target sequences and send to device"""

    source, targets, src_len = batch
    # pad source and target sequences
    source = pad_sequence(
        [torch.LongTensor(s) for s in source], batch_first=False, padding_value=0
    )
    targets = pad_sequence(
        [torch.LongTensor(t) for t in targets], batch_first=False, padding_value=0
    )
    src_len = torch.LongTensor(src_len)
    source, targets, src_len = source.to(device), targets.to(device), src_len.to(device)
    return source, targets, src_len


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
