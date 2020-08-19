import torch
import os
import gc
from utils import prep_batch

from torchnlp.metrics import get_moses_multi_bleu
from torchtext.data.metrics import bleu_score


def valid_step(
    model, source, src_len, targets, task, criterion, teacher_forcing,
):

    if task == "translation":
        output = model(source, src_len, targets, teacher_forcing)
    elif task == "classification":
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
    model, iterator, task, optimizer, criterion, device, teacher_forcing=None
):

    model.eval()
    epoch_loss = 0
    for batch in iterator:
        source, targets, src_len = prep_batch(batch, device)
        optimizer.zero_grad()
        loss = valid_step(
            model, source, src_len, targets, task, criterion, teacher_forcing
        )
        epoch_loss += loss

    return epoch_loss / len(iterator)


def get_bleu_score(corpus1, corpus2):
    return bleu_score(corpus1, corpus2)
    # return get_moses_multi_bleu(corpus1, corpus2)

def calc_accuracy(mdl, X, Y):
    # reduce/collapse the classification dimension according to max op
    # resulting in most likely label
    max_vals, max_indices = mdl(X).max(1)
    # assumes the first dimension is batch size
    n = max_indices.size(0)  # index 0 for extracting the # of elements
    # calulate acc (note .item() to do float division)
    acc = (max_indices == Y).sum().item() / n
    return acc

def accuracy_score(y_true, y_pred):
    y_pred = np.concatenate(tuple(y_pred))
    y_true = np.concatenate(tuple([[t for t in y] for y in y_true])).reshape(y_pred.shape)
    return (y_true == y_pred).sum() / float(len(y_true))

def categorical_accuracy(preds, tags, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (tags != tag_pad_idx).nonzero() #return non-zero, non-pad elements
    correct = max_preds[non_pad_elements].squeeze(1).eq(tags[non_pad_elements])
    return correct.sum() / torch.FloatTensor([tags[non_pad_elements].shape[0]])