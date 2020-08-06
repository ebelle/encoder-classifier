import os
import time
import math
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim

from lazy_dataset import LazyDataset
from custom_collate import sort_batch
from classifier import Classifier
from train import train_model
from evaluate import evaluate_model
from utils import random_init_weights, count_parameters, epoch_time
from multiple_optim import make_muliti_optim


def new_encoder_dict(prev_state_dict):

    new_state_dict = OrderedDict()

    for k, v in prev_state_dict.items():
        if "encoder" in k:
            # remove encoder from key name since we're adding this directly to the encoder
            new_k = k.replace("encoder.", "")
            # create new state dict for encoder
            new_state_dict[new_k] = v
    return new_state_dict


def get_prev_params(prev_state_dict):
    # gather parameters from pre-trained model

    emb_dim = prev_state_dict["encoder.enc_embedding.weight"].shape[1]
    hid_dim = prev_state_dict["encoder.enc_embedding.weight"].shape[1]

    # determine if previous model was bidirectional
    # if so, halve the hid_dim so it matches the previous model
    for k in prev_state_dict.keys():
        if "reverse" in k:
            bidirectional = True
        else:
            bidirectional = False
    if bidirectional == True:
        hid_dim = hid_dim / 2

    # determine number of layers in previous model
    for k in prev_state_dict.keys():
        if "l1" in k:
            num_layers = 2
        else:
            num_layers = 1

    return emb_dim, hid_dim, bidirectional, num_layers


def main(args):

    # use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create directory for saving models if it doesn't already exist
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    SRC = torch.load(os.path.join(args.translation_data_path, "src_vocab.pt"))
    TRG = torch.load(os.path.join(args.classifier_data_path, "trg_vocab.pt"))

    # gather parameters from the vocabulary
    input_dim = len(SRC.vocab)
    output_dim = len(TRG.vocab)
    pad_idx = SRC.vocab.stoi[SRC.pad_token]

    # build dictionary of parameters for the Dataloader
    dataloader_params = {
        "batch_size": args.batch_size,
        "collate_fn": sort_batch,
        "num_workers": args.num_workers,
        "shuffle": args.shuffle_batch,
        "pin_memory": True,
    }

    # create lazydataset and data loader
    training_set = LazyDataset(args.data_path, "train.tsv", SRC, TRG)
    train_iterator = torch.utils.data.DataLoader(training_set, **dataloader_params)

    # load pretrained-model
    prev_state_dict = torch.load(args.model_path)["model_state_dict"]

    emb_dim, hid_dim, bidirectional, num_layers = get_prev_params(prev_state_dict)

    new_state_dict = new_encoder_dict(prev_state_dict)

    model = Classifier(
        new_state_dict,
        args.freeze_encoder,
        input_dim,
        emb_dim,
        hid_dim,
        output_dim,
        num_layers,
        args.dropout,
        bidirectional,
        pad_idx,
    ).to(device)

    # optionally randomly initialize weights
    if args.random_init:
        model.apply(random_init_weights)

    print(model)
    print(f"The model has {count_parameters(model):,} trainable parameters")

    optimizer = optim.Adam(p for p in model.parameters() if p.requires_grad)
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    best_valid_loss = float("inf")

    # training
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train_model(
            model,
            train_iterator,
            task="classification",
            optimizer=optimizer,
            criterion=criterion,
            clip=args.clip,
            device=device,
            epoch=epoch,
            start_time=start_time,
            save_path=args.save_path,
            checkpoint=args.checkpoint,
        )

        # optionally validate
        if args.validate == True:

            valid_set = LazyDataset(args.data_path, "valid.tsv", SRC, TRG)
            valid_iterator = torch.utils.data.DataLoader(valid_set, **dataloader_params)

            valid_loss = evaluate_model(
                model,
                valid_iterator,
                optimizer=optimizer,
                criterion=criterion,
                task="classification",
                device=device,
            )

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            model_filename = os.path.join(args.save_path, f"model_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": valid_loss,
                },
                model_filename,
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                best_filename = os.path.join(args.save_path, f"best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": valid_loss,
                    },
                    best_filename,
                )

            print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(
                f"\t Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
            )
            print(
                f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}"
            )

        else:
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # save models each epoch
            model_filename = os.path.join(args.save_path, f"model_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": valid_loss,
                },
                model_filename,
            )

            print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(
                f"\t Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
            )


if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-vocab-path", help="folder where source vocab is stored"
    )
    parser.add_argument("--model-path", help="folder where pre-trained model is stored")
    parser.add_argument(
        "--data-path", help="folder where data and target vocab are stored"
    )
    parser.add_argument(
        "--save-path", help="folder for saving model and/or checkpoints"
    )
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--shuffle-batch", default=True, type=bool)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--clip", default=1.0, type=float)
    parser.add_argument(
        "--freeze-encoder",
        default=False,
        type=bool,
        help="optionally freeze encoder so it does not train",
    )
    parser.add_argument(
        "--random-init", default=True, type=bool, help="randomly initialize weights"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="learning rate for optimizer"
    )
    parser.add_argument("--checkpoint", type=int, help="save model every N batches")
    parser.add_argument(
        "--validate", default=True, type=bool, help="set to False to skip validation"
    )

    main(parser.parse_args())
