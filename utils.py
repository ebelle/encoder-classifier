import os
import glob
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence


def random_init_weights(model):
    """randomly initialize weights"""
    for name, p in model.named_parameters():
        if "weight" in name:
            if p.requires_grad:
                nn.init.normal_(p.data, mean=0, std=0.01)
        else:
            nn.init.constant_(p.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prep_batch(batch, device, pad_indices):
    """pad source and target sequences and send to device"""

    src_pad_idx = pad_indices[0]
    trg_pad_idx = pad_indices[1]
    source, targets, src_len = batch
    # pad source and target sequences
    source = pad_sequence(
        [torch.LongTensor(s) for s in source],
        batch_first=False,
        padding_value=src_pad_idx,
    ).to(device)
    targets = pad_sequence(
        [torch.LongTensor(t) for t in targets],
        batch_first=False,
        padding_value=trg_pad_idx,
    ).to(device)
    src_len = torch.LongTensor(src_len).to(device)
    return source, targets, src_len


def prep_eval_batch(batch, device, trg_pad_idx):
    """pad source sequence. target stays as indexes of lines in file"""

    source, trg_indices, src_len = batch
    # pad source and target sequences
    source = pad_sequence(
        [torch.LongTensor(s) for s in source],
        batch_first=False,
        padding_value=trg_pad_idx,
    ).to(device)

    src_len = torch.LongTensor(src_len).to(device)
    return source, trg_indices, src_len


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def sort_batch(batch):
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    source, targets, src_len = zip(*[(s, t, l) for (s, t, l) in batch])
    return source, targets, src_len


class MultipleOptimizer(object):
    def __init__(self, op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def return_optimizers(self):
        return self.optimizers

    def load_optimizers(self):
        for op in self.optimizers:
            print(op)


def make_muliti_optim(
    parameters, learning_rate, adam_state_dict=None, sparseadam_state_dict=None
):
    """build compound optimizer for adam and sparseadam"""

    params = []
    sparse_params = []
    for k, p in parameters:
        if p.requires_grad:
            if "embed" not in k:
                params.append(p)
            else:
                sparse_params.append(p)
    adam = optim.Adam(params, lr=learning_rate)
    sparse = optim.SparseAdam(sparse_params, lr=learning_rate)
    if adam_state_dict and sparseadam_state_dict:
        adam.load_state_dict(adam_state_dict)
        sparse.load_state_dict(sparseadam_state_dict)
    optimizer = MultipleOptimizer([adam, sparse])
    return optimizer


def get_prev_params(prev_state_dict):
    """gather parameters from pre-trained model"""

    # create dict to store paramaters
    prev_param_dict = {}

    prev_param_dict["emb_dim"] = prev_state_dict["encoder.enc_embedding.weight"].shape[
        1
    ]
    if "encoder.rnn.weight_hh_l0" in prev_state_dict:
        prev_param_dict["enc_hid_dim"] = prev_state_dict[
            "encoder.rnn.weight_hh_l0"
        ].shape[1]
    else:
        prev_param_dict["enc_hid_dim"] = None

    # for NMT
    if "decoder.rnn.weight_hh_l0" in prev_state_dict:
        prev_param_dict["dec_hid_dim"] = prev_state_dict[
            "decoder.rnn.weight_hh_l0"
        ].shape[1]
    else:
        prev_param_dict["dec_hid_dim"] = prev_state_dict[
            "decoder.hidden_layer1.weight"
        ].shape[0]

    # determine if previous model was bidirectional
    for k in prev_state_dict.keys():
        if "reverse" in k:
            prev_param_dict["bidirectional"] = True
            # stop once you've found the key you want
            break
        else:
            prev_param_dict["bidirectional"] = False

    # determine number of layers in previous model
    # TODO: fix in case of more than 2 layers
    for k in prev_state_dict.keys():
        if "l1" in k:
            prev_param_dict["enc_layers"] = 2
            # stop once you've found the key you want
            break
        else:
            prev_param_dict["enc_layers"] = 1

    # for tagger, check for 2nd layer in decoder
    if "decoder.hidden_layer2.weight" in prev_state_dict:
        prev_param_dict["dec_layers"] = 2
    else:
        prev_param_dict["dec_layers"] = 1

    return prev_param_dict


def process_line(line, vocab_field, init_eos=False):
    line = line.split()
    if init_eos:
        line = [vocab_field.vocab.init_token] + line + [vocab_field.vocab.eos_token]
    line = [vocab_field.vocab.stoi[t] for t in line]
    return line


def get_best_loss(models_folder):
    glob_folder = os.path.join(models_folder, "**/*.pt")
    losses = {}
    for model in glob.glob(glob_folder, recursive=True):
        loss = round(torch.load(model)["loss"], 4)
        losses[model] = loss
    print({k: v for k, v in sorted(losses.items(), key=lambda item: item[1])})
