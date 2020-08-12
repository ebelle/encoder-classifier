import os
import csv
import torch
import argparse

from classifier import Classifier
from utils import prep_batch, get_prev_params
from lazy_dataset import LazyDataset


def tag_sentence(
    source, src_len, src_field, model, device
):
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
    enc_dropout,dec_dropout = torch.load(args.model_path)["dropout"]

    prev_state_dict = prev_state_dict['model_state_dict']

    emb_dim, enc_hid_dim, dec_hid_dim, bidirectional, num_layers = get_prev_params(
        prev_state_dict)

    # arg is needed for the init, but the model is in eval mode anyways
    freeze_encoder = False

    # create model
    model = Classifier(
        prev_state_dict,
        freeze_encoder,
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

    model.load_state_dict(prev_state_dict)

    predictions = []
    targets = []
    for sample in test_set:
        source, target, src_len = sample
        pred = tag_sentence(
            source, src_len, target, SRC, TRG, model, device, max_len=55
        )
        # indices to string
        predictions.append(' '.join([TRG.vocab.itos[i] for i in pred]))
        targets.append(' '.join([TRG.vocab.itos[i] for i in target]))
    # optionally save results to file
    if args.save_file: 
        with open(args.save_file, "w") as sink:
            writer = csv.writer(sink,delimiter='\t')
            writer.writerows(zip(predictions,targets))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", help="folder where data and dictionaries are stored"
    )
    parser.add_argument("--pretrained-model", help="filename of pretrained model")
    parser.add_argument("--save-file",default=None, help="optional filename for saving translated sentences")

    main(parser.parse_args())
