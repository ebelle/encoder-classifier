import os
import csv
import torch
import argparse
from torchtext.data.metrics import bleu_score
from torchnlp.metrics import get_moses_multi_bleu

from lstm import Seq2Seq
from utils import prep_batch, get_prev_params
from lazy_dataset import LazyDataset


def translate_sentence(
    source, src_len, target, src_field, trg_field, model, device, max_len
):

    model.eval()

    source = torch.LongTensor(source).unsqueeze(1).to(device)
    target = torch.LongTensor(target).unsqueeze(1).to(device)
    src_len = torch.LongTensor(src_len)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(source, src_len, target)

    mask = model.create_mask(source)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(src_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(
                trg_tensor, hidden, cell, encoder_outputs, mask
            )

        pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    return trg_indexes


def main(args):

    # use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create directory for saving models if it doesn't already exist
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    SRC = torch.load(os.path.join(args.data_path, "src_vocab.pt"))
    TRG = torch.load(os.path.join(args.data_path, "trg_vocab.pt"))

    # gather parameters from the vocabulary
    input_dim = len(SRC.vocab)
    output_dim = len(TRG.vocab)
    pad_idx = SRC.vocab.stoi[SRC.pad_token]

    prev_state_dict = torch.load(args.pretrained_model)
    try:
        dropout = torch.load(args.model_path)["dropout"]
    # TODO: Remove this before final version
    except:
        dropout = 0.5

    emb_dim, hid_dim, bidirectional, num_layers = get_prev_params(prev_state_dict)

    # create model
    model = Seq2Seq(
        input_dim,
        emb_dim,
        hid_dim,
        output_dim,
        num_layers,
        dropout,
        bidirectional,
        pad_idx,
        device,
    ).to(device)

    model.load_state_dict(prev_state_dict)

    test_set = LazyDataset(args.data_path, "test.tsv", SRC, TRG, "translation")
    predictions = []
    targets = []
    for sample in test_set:
        source, target, src_len = sample
        pred = translate_sentence(
            source, src_len, target, SRC, TRG, model, device, max_len=55
        )
        # strip init and eos tokens
        pred = pred[1:-1]
        target = target[1:-1]
        # replace indices with tokens
        predictions.append(' '.join([TRG.vocab.itos[i] for i in pred]))
        targets.append(' '.join([TRG.vocab.itos[i] for i in target]))
    # optionally save results to file
    if args.save_file: 
        with open(args.save_file, "w") as sink:
            writer = csv.writer(sink,delimiter='\t')
            writer.writerows(zip(predictions,targets))
    # print bleu score
    print(bleu_score(predictions,targets))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", help="folder where data and dictionaries are stored"
    )
    parser.add_argument("--save-file", help="filename for saving translated sentences")
    parser.add_argument("--pretrained-model", help="filename of pretrained model")

    main(parser.parse_args())
