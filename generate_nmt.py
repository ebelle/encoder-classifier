import os
import csv
import torch
import argparse
import linecache
import numpy as np
import operator

from lstm import Seq2Seq
from lazy_dataset import LazyDataset
from bucket_sampler import BucketBatchSampler
from utils import prep_eval_batch, get_prev_params, sort_batch, epoch_time
from torchtext.data.metrics import bleu_score
import sacrebleu
import time
from queue import PriorityQueue
import functools



def preds_to_toks(preds, vocab_field, min_len=5):

    preds = [[vocab_field.vocab.itos[i] for i in x] for x in preds]

    final = []
    eos = vocab_field.eos_token
    for sentence in preds:
        try:
            eos_idx = sentence.index(eos)
            # clip eos and anything after eos
            sentence = sentence[:eos_idx]
            final.append(" ".join(sentence))
        # unless there is no eos
        except:
            # clip trailing unks
            while len(sentence) > min_len and sentence[-1] == "<unk>":
                del sentence[-1]
            final.append(" ".join(sentence))
    # indices to strings
    return final


def get_target(filepath, target_idx):
    """get target string from file"""
    line = linecache.getline(filepath, target_idx + 2)
    # split on tab and get second value, which is the target
    # strip newline and split on whitespace
    line = line.split("\t")[1].strip()
    return line


@functools.total_ordering
class BeamSearchNode(object):
    def __init__(self, hiddenstate, cellstate, previousNode, wordId, logProb, length):

        self.h = hiddenstate
        self.c = cellstate
        self.prevNode = previousNode
        self.word_id = wordId
        self.logp = logProb
        self.leng = length

    def free_up_space(self):
        # reduce memory usage
        del self.h, self.c
        self.word_id = self.word_id.item()

    def __eq__(self, other):
        return self.eval() == other.eval()

    def __ne__(self, other):
        return self.eval() != other.eval()

    def __lt__(self, other):
        return self.eval() < other.eval()

    def eval(self, alpha=1.0):
        reward = 0
        # optionally add function for shaping reward

        return -self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def beam_decode(source, src_len, trg_vocab, model, device, beam_width=5):

    # generate one sentence
    topk = 1
    decoded_batch = []

    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(source, src_len)
        encoder_outputs, hidden, cell = encoder_outputs.cpu(), hidden.cpu(), cell.cpu()
        mask = model.create_mask(source)

        batch_size = source.shape[1]
        trg_init = trg_vocab.vocab.stoi[trg_vocab.init_token]

        # decode one sentence at a time
        for idx in range(batch_size):

            # get hidden,cell,enc_output and mask for 1 sequence
            dec_hid = hidden[:, idx, :].contiguous().unsqueeze(1)
            dec_cell = cell[:, idx, :].contiguous().unsqueeze(1)
            single_output = encoder_outputs[:, idx, :].unsqueeze(1).to(device)
            single_mask = mask[idx, :].unsqueeze(0)

            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([[trg_init]])

            # Number of sentences to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node
            node = BeamSearchNode(
                hiddenstate=dec_hid,
                cellstate=dec_cell,
                previousNode=None,
                wordId=decoder_input,
                logProb=0,
                length=1,
            )
            nodes = PriorityQueue()

            # start the queue
            nodes.put(node)
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000:
                    break

                # fetch the best node
                n = nodes.get()
                decoder_input = n.word_id.squeeze(1).to(device)
                dec_hid = n.h.to(device)
                dec_cell = n.c.to(device)
                n.free_up_space()

                if n.word_id == trg_vocab.eos_token and n.prevNode != None:
                    endnodes.append(n)
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, dec_hid, dec_cell = model.decoder(
                    decoder_input, dec_hid, dec_cell, single_output, single_mask,
                )
                # beam search of top N where N is beam width
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []
                del decoder_output

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(
                        dec_hid.cpu(),
                        dec_cell.cpu(),
                        n,
                        decoded_t.cpu(),
                        n.logp + log_p,
                        n.leng + 1,
                    )
                    nextnodes.append(node)

                # put them into queue
                for i in range(len(nextnodes)):
                    nn = nextnodes[i]
                    nodes.put(nn)
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            preds = []
            for n in sorted(endnodes):
                single_pred = []
                single_pred.append(n.word_id)
                # back trace
                while n.prevNode:
                    n = n.prevNode
                    single_pred.append(n.word_id)

                single_pred = single_pred[::-1]
                # snip off init_token
                preds.append(single_pred[1:])

            decoded_batch.append(preds[0])
    return decoded_batch

def greedy_decode(source, src_len, trg_vocab, model, device, max_len=55):

    model.eval()

    with torch.no_grad():
        batch_size = source.shape[1]

        # tensor to store decoder outputs
        decoded_batch = torch.zeros((batch_size, max_len))

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden, cell = model.encoder(source, src_len)

        mask = model.create_mask(source)

        # free up memory
        del source, src_len
        # first input is the init_token
        trg_init = trg_vocab.vocab.stoi[trg_vocab.init_token]
        decoder_input = torch.LongTensor([[trg_init] for _ in range(batch_size)]).to(
            device
        )

        # [bsz,1] -> [bsz]
        decoder_input = decoder_input.squeeze(1)

        for t in range(max_len):

            decoder_output, hidden, cell = model.decoder(
                decoder_input, hidden, cell, encoder_outputs, mask
            )
            # get most likely predicted token
            # it will also be the input at the next step
            decoder_input = decoder_output.data.topk(1)[1].view(-1)
            # speed up processing by moving to cpu
            decoded_batch[:, t] = decoder_input.cpu()

    return decoded_batch

def main(args):

    # use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SRC = torch.load(os.path.join(args.data_path, "src_vocab.pt"))
    TRG = torch.load(os.path.join(args.data_path, "trg_vocab.pt"))

    # gather parameters from the vocabulary
    input_dim = len(SRC.vocab)
    output_dim = len(TRG.vocab)
    pad_idx = SRC.vocab.stoi[SRC.pad_token]

    model_dict = torch.load(args.pretrained_model)
    dropout = model_dict["dropout"]

    prev_state_dict = model_dict["model_state_dict"]
    del model_dict

    # gather parameters except dec_hid_dim since in this model they are the same
    prev_param_dict = get_prev_params(prev_state_dict)

    # create model
    model = Seq2Seq(
        input_dim,
        prev_param_dict["emb_dim"],
        prev_param_dict["enc_hid_dim"],
        output_dim,
        prev_param_dict["enc_layers"],
        dropout,
        prev_param_dict["bidirectional"],
        pad_idx,
        device,
    ).to(device)

    model.load_state_dict(prev_state_dict)
    del prev_state_dict
    print(model)

    test_path = os.path.join(args.data_path, "test.tsv")
    test_set = LazyDataset(test_path, SRC, TRG, "evaluation")

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

    final_preds = []
    final_targets = []
    for i, batch in enumerate(test_iterator):
        source, target_indicies, src_len = prep_eval_batch(batch, device,TRG.vocab.stoi[TRG.pad_token])
        # get targets from file
        final_targets += [get_target(test_path, idx) for idx in target_indicies]

        if args.decode_method == "beam":
            final_preds += preds_to_toks(
                beam_decode(source, src_len, TRG, model, device), TRG
            )

        elif args.decode_method == "greedy":
            preds = greedy_decode(source, src_len, TRG, model, device)
            # tensor to integer numpy array for quicker processing
            preds = preds.numpy().astype(int)
            final_preds += preds_to_toks(preds, TRG)

        if i % int(len(test_iterator)/100) == 0:
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f" batch {i} |Time: {epoch_mins}m {epoch_secs}s")
            start_time = end_time

    if args.save_file:
        sink = open(args.save_file, "w")
        writer = csv.writer(sink, delimiter="\t")
        writer.writerows(zip(final_preds, final_targets))

    if not args.no_bleu:
        final_preds = [p.split() for p in final_preds]
        final_targets = [[t.split()] for t in final_targets]
        print(bleu_score(final_preds, final_targets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", help="folder where data and dictionaries are stored"
    )
    parser.add_argument("--pretrained-model", help="filename of pretrained model")
    parser.add_argument("--batch-size", default=512, type=int, help="batch size")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument(
        "--decode-method",
        default="beam",
        choices=["greedy", "beam"],
        type=str,
        help="decoder method. choices are beam or greedy",
    )
    parser.add_argument(
        "--no-bleu",
        default=False,
        action='store_true',
        help="do not compute BLEU",
    )
    parser.add_argument(
        "--save-file",
        default=None,
        help="tsv filename for saving predictions translated sentences",
    )

    main(parser.parse_args())
