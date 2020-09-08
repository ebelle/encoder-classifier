import os
import time
import csv
import torch
import argparse
import linecache
from statistics import mean
import seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


from tagger import Tagger
from utils import prep_batch, get_prev_params, sort_batch, epoch_time
from lazy_dataset import LazyDataset
from bucket_sampler import BucketBatchSampler

from sklearn.metrics import classification_report, confusion_matrix


def model_forward(source, src_len, model, device):
    model.eval()
    with torch.no_grad():
        encoder_outputs = model.encoder(source, src_len)
        preds = model.decoder(encoder_outputs)
    return preds.cpu()


def top_predictions(preds):
    # gets predictions and clips padding
    batch_size = preds.shape[1]
    # get index of top prediction along the last dimension and resize to [seq_len,bsz]
    max_preds = torch.topk(preds, k=1, dim=2)[1].view(-1, batch_size)
    return max_preds


def get_accuracy(preds, tags, target_lenth):
    # the overlap between predictions and tags
    correct = preds.eq(tags)
    # get the sum of correct predictions as an integer
    correct = correct.sum().item()
    percent_correct = correct / target_lenth
    return percent_correct


def categorical_accuracy(source, tags, preds, src_unk_idx, tag_pad_idx):
    # Returns percent correct per batch
    # non-pad elements
    non_pad_elements = torch.nonzero(tags != tag_pad_idx, as_tuple=True)
    preds = preds[non_pad_elements]
    tags = tags[non_pad_elements]
    source = source[non_pad_elements]
    total_targets = len(non_pad_elements[0])
    non_pad_accuracy = get_accuracy(preds, tags, total_targets)

    # non zero elements
    non_zero_elements = torch.nonzero(source != src_unk_idx, as_tuple=True)
    known_preds = preds[non_zero_elements]
    known_tags = tags[non_zero_elements]
    known_source = source[non_zero_elements]
    total_known = len(non_zero_elements[0])
    known_accuracy = get_accuracy(known_preds, known_tags, total_known)

    # zero elements
    zero_elements = torch.nonzero(source == src_unk_idx, as_tuple=True)
    unk_preds = preds[zero_elements]
    unk_tags = tags[zero_elements]
    unk_source = source[zero_elements]
    total_unk = len(zero_elements[0])
    # return string in case of no unknowns
    unknown_accuracy = 'none'
    if total_unk:
        unknown_accuracy = get_accuracy(unk_preds, unk_tags, total_unk)

    return (
        non_pad_accuracy,
        known_accuracy,
        unknown_accuracy,
        preds,
        tags,
        known_preds,
        known_tags,
        unk_preds,
        unk_tags,
    )


def get_tag_dict(target_vocab):
    names = {}
    for i in range(len(target_vocab.vocab)):
        names[i] = target_vocab.vocab.itos[i]
    return names


def idx_to_tag(batch, src_len, vocabulary):
    """takes in batch a string of tokens"""

    # make horizontal
    batch = np.rot90(batch)
    # cast tensor to list and reverse it to match the batch
    src_len = src_len.tolist()
    src_len.reverse()
    # clip padding
    batch = [b[:l] for b, l in zip(batch, src_len)]
    # get tokens and join into string
    batch = [" ".join(vocabulary.vocab.itos[i] for i in x) for x in batch]
    return batch


def make_confusion_matrix(preds, targs, tag_names, save_path, pic_file):
    confusion_matrix_df = pd.DataFrame(confusion_matrix(targs, preds)).rename(
        columns=tag_names, index=tag_names
    )
    cm = seaborn.heatmap(confusion_matrix_df)
    cm.set_xlabel("Predicted")
    cm.set_ylabel("True")
    plt.tight_layout()
    fig = cm.get_figure()
    fig.savefig(os.path.join(save_path, pic_file))
    del fig, cm, confusion_matrix_df


def main(args):

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SRC = torch.load(os.path.join(args.nmt_data_path, "src_vocab.pt"))
    TRG = torch.load(os.path.join(args.data_path, "trg_vocab.pt"))

    # gather parameters from the vocabulary
    input_dim = len(SRC.vocab)
    output_dim = len(TRG.vocab)
    src_unk_idx = SRC.vocab.stoi[SRC.unk_token]
    src_pad_idx = SRC.vocab.stoi[SRC.pad_token]
    trg_pad_idx = len(TRG.vocab) + 1

    prev_state_dict = torch.load(
        args.pretrained_model, map_location=torch.device("cpu")
    )
    enc_dropout, dec_dropout = prev_state_dict["dropout"]
    repr_layer = prev_state_dict["repr_layer"]

    prev_state_dict = prev_state_dict["model_state_dict"]

    prev_param_dict = get_prev_params(prev_state_dict)

    # create model
    model = Tagger(
        input_dim,
        prev_param_dict["emb_dim"],
        prev_param_dict["enc_hid_dim"],
        prev_param_dict["dec_hid_dim"],
        output_dim,
        prev_param_dict["enc_layers"],
        prev_param_dict["dec_layers"],
        enc_dropout,
        dec_dropout,
        prev_param_dict["bidirectional"],
        src_pad_idx,
        repr_layer,
    ).to(device)

    model.load_state_dict(prev_state_dict)

    test_path = os.path.join(args.data_path, "test.tsv")
    test_set = LazyDataset(test_path, SRC, TRG, "tagging")

    test_batch_sampler = BucketBatchSampler(test_path, args.batch_size)
    num_batches = test_batch_sampler.num_batches
    ten_checkpoints = np.linspace(0, num_batches, num=10, dtype=int)
    # build dictionary of parameters for the Dataloader
    test_loader_params = {
        # since bucket sampler returns batch, batch_size is 1
        "batch_size": 1,
        # sort_batch reverse sorts for pack_pad_seq
        "collate_fn": sort_batch,
        "batch_sampler": test_batch_sampler,
        "num_workers": args.num_workers,
        "shuffle": False,
        "pin_memory": True,
        "drop_last": False,
    }

    test_iterator = torch.utils.data.DataLoader(test_set, **test_loader_params)

    start_time = time.time()

    non_pad_accuracy = []
    known_accuracy = []
    unknown_accuracy = []
    final_predictons = []
    final_targets = []
    known_predictions = []
    known_targets = []
    unknown_predictions = []
    unknown_targets = []

    if args.save_path:
        # optionally save results to file
        sink = open(os.path.join(args.save_path, "tagged.tsv"), "w", encoding="utf-8")
        writer = csv.writer(sink, delimiter="\t")

    for i, batch in enumerate(test_iterator):
        source, targets, src_len = prep_batch(batch, device, (src_pad_idx, trg_pad_idx))
        # targets don't need to be on the gpu
        targets = targets.cpu()
        predictions = model_forward(source, src_len, model, device)
        predictions = top_predictions(predictions)
        source = source.cpu()
        if args.save_path:
            tok_preds = idx_to_tag(predictions, src_len, TRG)
            tok_trgs = idx_to_tag(targets, src_len, TRG)
            tok_source = idx_to_tag(source, src_len, SRC)
            writer.writerows(zip(tok_source, tok_trgs, tok_preds))
        (
            non_pad,
            known,
            unknown,
            predictions,
            targets,
            known_preds,
            known_tags,
            unk_preds,
            unk_tags,
        ) = categorical_accuracy(source, targets, predictions, src_unk_idx, trg_pad_idx)
        non_pad_accuracy.append(non_pad)
        known_accuracy.append(known)
        # we return a string if there are no unknowns
        if isinstance(unknown,str):
            pass
        else:
            unknown_accuracy.append(unknown)
        final_predictons += predictions
        final_targets += targets
        known_predictions += known_preds
        known_targets += known_tags
        unknown_predictions += unk_preds
        unknown_targets += unk_tags
    print(
        f"Model: {args.pretrained_model} | total accuracy: {mean(non_pad_accuracy):.4f} | known accuracy: {mean(known_accuracy):.4f} | unknown accuracy: {mean(unknown_accuracy):.4f}"
    )
    tag_names = get_tag_dict(TRG)
    # unknown cf
    """make_confusion_matrix(
        unknown_predictions,
        unknown_targets,
        tag_names,
        args.save_path,
        "unknown_matrix.png",
    )"""
    # total cf
    make_confusion_matrix(
        final_predictons,
        final_targets,
        tag_names,
        args.save_path,
        "confusion_matrix.png",
    )
    # known cf

    """make_confusion_matrix(
        known_predictions, known_targets, tag_names, args.save_path, "known_matrix.png"
    )"""
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmt-data-path", help="folder where source vocab is stored")
    parser.add_argument(
        "--data-path", help="folder where data and dictionaries are stored"
    )
    parser.add_argument("--batch-size", default=512, type=int, help="batch size")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--pretrained-model", help="filename of pretrained model")
    parser.add_argument(
        "--save-path", default=None, help="optional folder for saving output",
    )
    main(parser.parse_args())
