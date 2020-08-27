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


def categorical_accuracy(preds, tags, tag_pad_idx, tag_unk_idx):
    # Returns percent correct per batch
    # return non-zero, non-pad elements
    non_pad_elements = torch.nonzero(tags != tag_pad_idx or tag_unk_idx, as_tuple=True)
    preds = preds[non_pad_elements]
    tags = tags[non_pad_elements]
    # the overlap between predictions and tags
    correct = preds.eq(tags)
    # get the sum of correct predictions as an integer
    correct = correct.sum().item()
    total_nonzero_targets = tags.shape[0]
    percent_correct = correct / total_nonzero_targets

    return percent_correct, preds, tags


def get_tag_dict(target_vocab):
    names = {}
    # skip the pad token
    for i in range(1, len(target_vocab.vocab)):
        names[i] = target_vocab.vocab.itos[i]
    return names


def idx_to_tag(preds, source, trg_vocab, src_vocab, trg_pad_idx, src_pad_idx):
    # make source horizontal
    source = np.rot90(source)
    preds = np.rot90(preds)
    preds = [[trg_vocab.vocab.itos[i] for i in x] for x in preds]
    source = [[src_vocab.vocab.itos[i] for i in x] for x in source]

    return preds, source


def main(args):

    # use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SRC = torch.load(os.path.join(args.nmt_data_path, "src_vocab.pt"))
    TRG = torch.load(os.path.join(args.data_path, "trg_vocab.pt"))

    # gather parameters from the vocabulary
    input_dim = len(SRC.vocab)
    output_dim = len(TRG.vocab)
    src_pad_idx = SRC.vocab.stoi[SRC.pad_token]
    trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]
    trg_unk_idx = TRG.vocab.stoi[TRG.unk_index]

    prev_state_dict = torch.load(args.pretrained_model)
    enc_dropout, dec_dropout = torch.load(args.pretrained_model)["dropout"]

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
    ).to(device)

    model.load_state_dict(prev_state_dict)

    test_path = os.path.join(args.data_path, "test.tsv")
    test_set = LazyDataset(test_path, SRC, TRG, "tagging")

    test_batch_sampler = BucketBatchSampler(test_path, args.batch_size)

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

    accuracies = []
    final_predictons = []
    final_targets = []
    if args.save_file:
        # optionally save results to file
        sink = open(args.save_file, "w")
        writer = csv.writer(sink, delimiter="\t")
        print(len(test_iterator))
    for i, batch in enumerate(test_iterator):
        source, targets, src_len = prep_batch(batch, device)
        # targets don't need to be on the gpu
        targets = targets.cpu()
        predictions = model_forward(source, src_len, model, device)
        predictions = top_predictions(predictions)
        if args.save_file:
            source = source.cpu()
            tok_preds, tok_source = idx_to_tag(
                predictions, source, TRG, SRC, trg_pad_idx, src_pad_idx
            )
            writer.writerows(zip(tok_source, tok_preds))
        acc, predictions, targets = categorical_accuracy(
            predictions, targets, trg_pad_idx, trg_unk_idx
        )
        accuracies.append(acc)
        final_predictons += predictions
        final_targets += targets

        if i % 10000 == 0:
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f" batch {i} |Time: {epoch_mins}m {epoch_secs}s")
            start_time = end_time
    print(mean(accuracies))
    print(classification_report(final_predictons, final_targets))

    tag_names = get_tag_dict(TRG)
    confusion_matrix_df = pd.DataFrame(
        confusion_matrix(final_predictons, final_predictons)
    ).rename(columns=tag_names, index=tag_names)
    seaborn.heatmap(confusion_matrix_df)
    plt.show()


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
        "--save-file",
        default=None,
        help="optional filename for saving translated sentences",
    )

    main(parser.parse_args())
