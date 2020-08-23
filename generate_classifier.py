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


from classifier import Classifier
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


def categorical_accuracy(preds, tags, tag_pad_idx):
    # Returns percent correct per batch
    batch_size = tags.shape[1]
    # get index of top prediction along the last dimension and resize to [seq_len,bsz]
    max_preds = torch.topk(preds, k=1, dim=2)[1].view(-1, batch_size)
     # return non-zero, non-pad elements
    non_pad_elements = torch.nonzero(
        tags != tag_pad_idx, as_tuple=True
    )
    max_preds = max_preds[non_pad_elements]
    tags = tags[non_pad_elements]
    # the overlap between predictions and tags
    correct = max_preds.eq(tags)
    # get the sum of correct predictions as an integer
    correct = correct.sum().item()
    total_nonzero_targets = tags.shape[0]
    percent_correct = correct / total_nonzero_targets

    return percent_correct, max_preds, tags


def get_tag_dict(target_vocab):
    names = {}
    # skip the pad token
    for i in range(1,len(target_vocab.vocab)):
        names[i] = target_vocab.vocab.itos[i]
    return names


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

    prev_state_dict = torch.load(args.pretrained_model)
    enc_dropout, dec_dropout = torch.load(args.pretrained_model)["dropout"]

    prev_state_dict = prev_state_dict["model_state_dict"]

    emb_dim, enc_hid_dim, dec_hid_dim, bidirectional, num_layers = get_prev_params(
        prev_state_dict
    )

    # create model
    model = Classifier(
        input_dim,
        emb_dim,
        enc_hid_dim,
        dec_hid_dim,
        output_dim,
        num_layers,
        enc_dropout,
        dec_dropout,
        bidirectional,
        src_pad_idx,
    ).to(device)

    model.load_state_dict(prev_state_dict)

    test_path = os.path.join(args.data_path, "test.tsv")
    test_set = LazyDataset(test_path, SRC, TRG, "classification")

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
    for i, batch in enumerate(test_iterator):
        source, targets, src_len = prep_batch(batch, device)
        # targets don't need to be on the gpu
        targets = targets.cpu()
        predictions = model_forward(source, src_len, model, device)
        acc, preds, targs = categorical_accuracy(predictions, targets, trg_pad_idx)
        accuracies.append(acc)
        final_predictons += preds
        final_targets += targs
        if i % 5 == 0:
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
    
    # optionally save results to file
    with open(args.save_file, "w") as sink:
        writer = csv.writer(sink, delimiter="\t")
        writer.writerows(zip(predictions, targets))


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
