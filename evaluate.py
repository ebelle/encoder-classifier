import torch
import os
import gc
from utils import prep_batch


def valid_step(
    model, source, src_len, targets, task, criterion, teacher_forcing,
):

    if task == "translation":
        output = model(source, src_len, targets, teacher_forcing)
    elif task == "tagging":
        output = model(source, src_len)

    # trg = [trg_len, bsz]
    # output = [src_len, bsz, output dim]

    output_dim = output.shape[-1]

    output = output[1:].view(-1, output_dim)
    targets = targets[1:].view(-1)

    # trg = [(trg len - 1) * bsz]
    # output = [(trg len - 1) * bsz, output dim]

    # return loss
    return criterion(output, targets).item()


def evaluate_model(
    model,
    iterator,
    task,
    optimizer,
    criterion,
    device,
    pad_indices,
    num_batches,
    teacher_forcing=None,
):

    model.eval()
    epoch_loss = 0
    for batch in iterator:
        source, targets, src_len = prep_batch(batch, device, pad_indices)
        optimizer.zero_grad()
        loss = valid_step(
            model, source, src_len, targets, task, criterion, teacher_forcing
        )
        epoch_loss += loss

    return epoch_loss / num_batches
