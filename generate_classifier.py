import os
import csv
import torch
import argparse
import linecache

from classifier import Classifier
from utils import prep_batch, get_prev_params, process_line
from lazy_dataset import LazyDataset
from evaluate import categorical_accuracy


def tag_sentence(source, src_len, src_field, model, device):
    model.eval()

    source = torch.LongTensor(source).unsqueeze(1).to(device)
    src_len = torch.LongTensor(src_len)

    with torch.no_grad():
        encoder_outputs = model.encoder(source, src_len)
        tags = model.decoder(encoder_outputs)
    return tags


def main(args):

    # use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SRC = torch.load(os.path.join(args.data_path, "src_vocab.pt"))
    TRG = torch.load(os.path.join(args.data_path, "trg_vocab.pt"))

    # gather parameters from the vocabulary
    input_dim = len(SRC.vocab)
    output_dim = len(TRG.vocab)
    pad_idx = SRC.vocab.stoi[SRC.pad_token]

    prev_state_dict = torch.load(args.pretrained_model)
    enc_dropout, dec_dropout = torch.load(args.model_path)["dropout"]

    prev_state_dict = prev_state_dict["model_state_dict"]

    emb_dim, enc_hid_dim, dec_hid_dim, bidirectional, num_layers = get_prev_params(
        prev_state_dict
    )

    # create model
    model = Classifier(
        prev_state_dict,
        input_dim,
        emb_dim,
        enc_hid_dim,
        dec_hid_dim,
        output_dim,
        num_layers,
        enc_dropout,
        dec_dropout,
        bidirectional,
        pad_idx,
    ).to(device)

    test_path = os.path.join(args.data_path, "test.tsv")
    total_data = sum(1 for _ in open(test_path, "r"))

    predictions = []
    targets = []
    for i in range(total_data):
        line = linecache.getline(test_path, i + 1)
        source, target, src_len = process_line(line, SRC)
        targets.append(target)
        pred = tag_sentence(source, src_len, SRC, model, device)
        # indices to string
        predictions.append(" ".join([TRG.vocab.itos[i] for i in pred]))
    # optionally save results to file
    with open(args.save_file, "w") as sink:
        writer = csv.writer(sink, delimiter="\t")
        writer.writerows(zip(predictions, targets))
    #print(categorical_accuracy(predictions, targets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", help="folder where data and dictionaries are stored"
    )
    parser.add_argument("--pretrained-model", help="filename of pretrained model")
    parser.add_argument(
        "--save-file",
        default=None,
        help="optional filename for saving translated sentences",
    )

    main(parser.parse_args())
