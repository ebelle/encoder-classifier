import torch


def sort_batch(batch):
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    source, targets, src_len = zip(*[(s, t, l) for (s, t, l) in batch])

    return source, targets, src_len
