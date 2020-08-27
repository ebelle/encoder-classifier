import os
import time
import math
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim

import matplotlib.pyplot as plt

from lazy_dataset import LazyDataset
from bucket_sampler import BucketBatchSampler
from tagger import Tagger
from train import train_model
from evaluate import evaluate_model
from utils import (
    random_init_weights,
    count_parameters,
    epoch_time,
    sort_batch,
    get_prev_params,
    make_muliti_optim,
)


def remove_encoder_str(prev_state_dict):

    new_state_dict = OrderedDict()

    for k, v in prev_state_dict.items():
        if "encoder" in k:
            # remove encoder from key name since we're adding this directly to the encoder
            new_k = k.replace("encoder.", "")
            # create new state dict for encoder
            new_state_dict[new_k] = v
    return new_state_dict


def make_loss_plot(model_history):
    ax = plt.subplot(111)
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.plot(
        list(range(1, len(model_history) + 1)), model_history, label="training loss"
    )
    plt.xlabel("batch", fontsize=16)
    plt.ylabel("training loss", fontsize=14)
    ax.set_title("Training Loss", fontsize=20, pad=40)
    plt.xticks(list(range(100, len(model_history) + 1, 100)))
    plt.legend()
    plt.show()


def main(args):

    # use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create directory for saving models if it doesn't already exist
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    SRC = torch.load(os.path.join(args.nmt_data_path, "src_vocab.pt"))
    TRG = torch.load(os.path.join(args.data_path, "trg_vocab.pt"))

    # gather parameters from the vocabulary
    input_dim = len(SRC.vocab)
    output_dim = len(TRG.vocab)
    pad_idx = SRC.vocab.stoi[SRC.pad_token]

    # create lazydataset and data loader
    train_path = os.path.join(args.data_path, "train.tsv")
    training_set = LazyDataset(train_path, SRC, TRG, "tagging")

    train_batch_sampler = BucketBatchSampler(train_path, args.batch_size)

    # build dictionary of parameters for the Dataloader
    train_loader_params = {
        # since bucket sampler returns batch, batch_size is 1
        "batch_size": 1,
        # sort_batch reverse sorts for pack_pad_seq
        "collate_fn": sort_batch,
        "batch_sampler": train_batch_sampler,
        "num_workers": args.num_workers,
        "shuffle": args.shuffle,
        "pin_memory": True,
        "drop_last": False,
    }

    train_iterator = torch.utils.data.DataLoader(training_set, **train_loader_params)

    # load pretrained-model
    prev_state_dict = torch.load(args.pretrained_model)["model_state_dict"]
    enc_dropout = torch.load(args.pretrained_model)["dropout"]

    # gather parameters except dec_hid_dim since tagger gets this from args
    prev_param_dict = get_prev_params(prev_state_dict)

    new_state_dict = remove_encoder_str(prev_state_dict)

    model = Tagger(
        new_state_dict=new_state_dict,
        input_dim=input_dim,
        emb_dim=prev_param_dict["emb_dim"],
        enc_hid_dim=prev_param_dict["enc_hid_dim"],
        dec_hid_dim=args.tag_hid_dim,
        output_dim=output_dim,
        enc_layers=prev_param_dict["enc_layers"],
        dec_layers=args.tag_layers,
        enc_dropout=enc_dropout,
        dec_dropout=args.tag_dropout,
        bidirectional=prev_param_dict["bidirectional"],
        pad_idx=pad_idx,
    ).to(device)

    # optionally randomly initialize weights
    if args.random_init:
        model.apply(random_init_weights)

    print(model)
    print(f"The model has {count_parameters(model):,} trainable parameters")

    optimizer = make_muliti_optim(model.named_parameters(), args.learning_rate)

    if args.unfreeze_encoder == False:
        for param in model.encoder.parameters():
            param.requires_grad = False

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    best_valid_loss = float("inf")

    # training
    batch_history = []
    epoch_history = []
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, batch_loss = train_model(
            model=model,
            iterator=train_iterator,
            task="tagging",
            optimizer=optimizer,
            criterion=criterion,
            clip=args.clip,
            device=device,
            epoch=epoch,
            start_time=start_time,
            save_path=args.save_path,
            dropout=(enc_dropout, args.tag_dropout),
            checkpoint=args.checkpoint,
        )
        batch_history += batch_loss
        epoch_history.append(train_loss)

        # optionally validate
        if not args.skip_validate:
            valid_path = os.path.join(args.data_path, "valid.tsv")
            valid_set = LazyDataset(valid_path, SRC, TRG, "tagging")
            valid_batch_sampler = BucketBatchSampler(valid_path, args.batch_size)
            valid_loader_params = {
                # since bucket sampler returns batch, batch_size is 1
                "batch_size": 1,
                # sort_batch reverse sorts for pack_pad_seq
                "collate_fn": sort_batch,
                "batch_sampler": valid_batch_sampler,
                "num_workers": args.num_workers,
                "shuffle": args.shuffle,
                "pin_memory": True,
                "drop_last": False,
            }

            valid_iterator = torch.utils.data.DataLoader(
                valid_set, **valid_loader_params
            )

            valid_loss = evaluate_model(
                model,
                valid_iterator,
                optimizer=optimizer,
                criterion=criterion,
                task="tagging",
                device=device,
            )

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            model_filename = os.path.join(args.save_path, f"model_epoch_{epoch}.pt")
            adam, sparse_adam = optimizer.return_optimizers()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "adam_state_dict": adam.state_dict(),
                    "sparse_adam_state_dict": sparse_adam.state_dict(),
                    "loss": valid_loss,
                    "dropout": (enc_dropout, args.tag_dropout),
                },
                model_filename,
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                best_filename = os.path.join(args.save_path, f"best_model.pt")
                adam, sparse_adam = optimizer.return_optimizers()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "adam_state_dict": adam.state_dict(),
                        "sparse_adam_state_dict": sparse_adam.state_dict(),
                        "loss": valid_loss,
                        "dropout": (enc_dropout, args.tag_dropout),
                    },
                    best_filename,
                )

            print(f"Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s")
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
            adam, sparse_adam = optimizer.return_optimizers()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "adam_state_dict": adam.state_dict(),
                    "sparse_adam_state_dict": sparse_adam.state_dict(),
                    "loss": train_loss,
                    "dropout": (enc_dropout, args.cls_dropout),
                },
                model_filename,
            )

            print(f"Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(
                f"\t Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
            )

    if args.loss_plot:
        make_loss_plot(batch_history)


if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmt-data-path", help="folder where source vocab is stored")
    parser.add_argument(
        "--pretrained-model", help="folder where pre-trained model is stored"
    )
    parser.add_argument(
        "--data-path", help="folder where data and target vocab are stored"
    )
    parser.add_argument(
        "--save-path", help="folder for saving model and/or checkpoints"
    )
    parser.add_argument(
        "--skip-validate",
        default=False,
        action="store_true",
        help="set to False to skip validation",
    )
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument(
        "--tag-layers",
        default=1,
        type=int,
        choices=[1, 2],
        help="number of tagger layers. options are 1 or 2",
    )
    parser.add_argument("--tag-dropout", default=0.1, type=float)
    parser.add_argument("--clip", default=1.0, type=float)
    parser.add_argument(
        "--tag-hid-dim", default=512, type=int, help="hidden dimension for tagger",
    )
    parser.add_argument(
        "--unfreeze-encoder",
        default=False,
        action="store_true",
        help="optionally freeze encoder so it does not train",
    )
    parser.add_argument(
        "--random-init",
        default=False,
        action="store_true",
        help="randomly initialize weights",
    )
    parser.add_argument(
        "--shuffle", default=False, action="store_true", help="shuffle batch",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="learning rate for optimizer"
    )
    parser.add_argument("--checkpoint", type=int, help="save model every N batches")
    parser.add_argument(
        "--loss-plot", default=False, action="store_true", help="create a loss plot"
    )
    main(parser.parse_args())
