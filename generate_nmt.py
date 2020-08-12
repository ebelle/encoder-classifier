import os
import csv
import torch
import argparse
import linecache

from lstm import Seq2Seq
from utils import prep_batch, get_prev_params
from evaluate import get_bleu_score


def translate_sentence(
    source, src_len, trg_field, model, device, max_len
):

    model.eval()

    # batch size of 1
    source = torch.LongTensor(source).unsqueeze(1).to(device)
    src_len = torch.LongTensor(src_len)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(source, src_len)

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

def process_line(line,SRC,init_token,eos_token,device):
    source,target = line.split("\t")
    source = source.split()
    source = [init_token] + source + [eos_token]
    source = [SRC.vocab.stoi[t] for t in source]
    src_len = [len(source)]
    return source, target, src_len

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
    try:
        dropout = torch.load(args.model_path)["dropout"]
    # TODO: Remove this before final version
    except:
        dropout = 0.5

    prev_state_dict = prev_state_dict['model_state_dict']

    # gather parameters except dec_hid_dim since in this model they are the same
    emb_dim, hid_dim, _, bidirectional, num_layers = get_prev_params(prev_state_dict)

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
    print(model)

    filepath = os.path.join(args.data_path, "test.tsv")
    total_data = sum(1 for _ in open(filepath, "r"))

    predictions = []
    targets = []
    init_token = SRC.init_token
    eos_token = SRC.eos_token
    print(total_data)
    for i in range(total_data):
        if i % 1000 == True:
            print(i)
        line = linecache.getline(filepath, i + 1)
        source,target,src_len = process_line(line,SRC,init_token,eos_token,device)
    
        pred = translate_sentence(
            source, src_len, TRG, model, device, max_len=55
        )
        # strip init and eos tokens
        pred = pred[1:-1]
        # indices to string
        pred = ' '.join([TRG.vocab.itos[i] for i in pred])
        predictions.append(pred)
        targets.append(target)
    # optionally save results to file
    if args.save_file: 
        with open(args.save_file, "w") as sink:
            writer = csv.writer(sink,delimiter='\t')
            writer.writerows(zip(predictions,targets))
    # print bleu score
    print(get_bleu_score(predictions,targets))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", help="folder where data and dictionaries are stored"
    )
    parser.add_argument("--pretrained-model", help="filename of pretrained model")
    parser.add_argument("--save-file",default=None, help="optional filename for saving translated sentences")

    main(parser.parse_args())
