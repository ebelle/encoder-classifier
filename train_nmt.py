import os
import time
import math
import argparse

import torch
import torch.nn as nn
from torch import optim

from lazy_dataset import LazyDataset
from lstm import Seq2Seq
from train import train_model
from evaluate import evaluate_model
from utils import *
from bucket_sampler import BucketBatchSampler


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
    src_pad_idx = SRC.vocab.stoi[SRC.pad_token]

    # create lazydataset and data loader
    train_path = os.path.join(args.data_path, "train.tsv")
    training_set = LazyDataset(train_path, SRC, TRG, "translation")

    bucket_batch_sampler = BucketBatchSampler(train_path, args.batch_size)

    # build dictionary of parameters for the Dataloader
    dataloader_params = {
        # since bucket sampler returns batch, batch_size is 1
        "batch_size": 1,
        # sort_batch reverse sorts for pack_pad_seq
        "collate_fn": sort_batch,
        "batch_sampler": bucket_batch_sampler,
        "num_workers": args.num_workers,
        "shuffle": args.shuffle,
        "pin_memory": True,
        "drop_last": False,
    }

    train_iterator = torch.utils.data.DataLoader(training_set, **dataloader_params)

    if not args.continue_training_model:
        # create model
        model = Seq2Seq(
            input_dim,
            args.embedding_dim,
            args.hidden_size,
            output_dim,
            args.num_layers,
            args.dropout,
            args.bidirectional,
            src_pad_idx,
            device,
        ).to(device)

        # optionally randomly initialize weights
        if args.random_init:
            model.apply(random_init_weights)

        # optionally freeze pretrained embeddings
        if args.freeze_embeddings:
            try:
                src_pretrained_embeddings = SRC.vocab.vectors
                model.encoder.enc_embedding.weight.data.copy_(src_pretrained_embeddings)
                model.encoder.enc_embedding.weight.requires_grad = False
            except TypeError:
                print(
                    "Cannot freeze embedding layer without pretrained embeddings. Rerun make_vocab with source vectors"
                )
        start_epoch = 1
        optimizer = make_muliti_optim(model.named_parameters(), args.learning_rate)
        best_valid_loss = float("inf")

    else:
        model_dict = torch.load(args.continue_training_model)
        prev_state_dict = model_dict["model_state_dict"]
        emb_dim, hid_dim, _, bidirectional, num_layers = get_prev_params(
            prev_state_dict
        )
        dropout = model_dict["dropout"]
        model = Seq2Seq(
            input_dim,
            emb_dim,
            hid_dim,
            output_dim,
            num_layers,
            dropout,
            bidirectional,
            src_pad_idx,
            device,
        ).to(device)

        start_epoch = model_dict["epoch"]

        adam_state_dict = model_dict["adam_state_dict"]
        sparse_adam_state_dict = model_dict["sparse_adam_state_dict"]
        optimizer = make_muliti_optim(
            model.named_parameters(),
            args.learning_rate,
            adam_state_dict,
            sparse_adam_state_dict,
        )
        model.load_state_dict(prev_state_dict)
        best_valid_loss = model["loss"]
        del model_dict, prev_state_dict, adam_state_dict, sparse_adam_state_dict

    print(model)
    print(f"The model has {count_parameters(model):,} trainable parameters")

    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    # training
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        train_loss = train_model(
            model,
            train_iterator,
            task="translation",
            optimizer=optimizer,
            criterion=criterion,
            clip=args.clip,
            device=device,
            epoch=epoch,
            start_time=start_time,
            save_path=args.save_path,
            dropout=args.dropout,
            teacher_forcing=args.teacher_forcing,
            checkpoint=args.checkpoint,
        )

        # optionally validate
        if not args.skip_validate:

            valid_path = os.path.join(args.data_path, "valid.tsv")
            valid_set = LazyDataset(valid_path, SRC, TRG, "translation")
            bucket_batch_sampler = BucketBatchSampler(valid_path, args.batch_size)
            valid_iterator = torch.utils.data.DataLoader(valid_set, **dataloader_params)

            valid_loss = evaluate_model(
                model,
                valid_iterator,
                task="translation",
                optimizer=optimizer,
                criterion=criterion,
                teacher_forcing=args.teacher_forcing,
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
                    "dropout": args.dropout,
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
                        "adam_state_dict": adam.state_dict(),
                        "sparse_adam_state_dict": sparse_adam.state_dict(),
                        "loss": valid_loss,
                        "dropout": args.dropout,
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
                    "dropout": args.dropout,
                },
                model_filename,
            )

            print(f"Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(
                f"\t Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", help="folder where data and dictionaries are stored"
    )
    parser.add_argument(
        "--save-path", help="folder for saving model and/or checkpoints"
    )
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--shuffle", default=False, action="store_true")
    parser.add_argument(
        "--random-init",
        default=False,
        action="store_true",
        help="randomly initialize weights",
    )
    parser.add_argument("--embedding-dim", default=300, type=int)
    parser.add_argument("--hidden-size", default=512, type=int)
    parser.add_argument("--num-layers", default=1, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--bidirectional", default=False, action="store_true")
    parser.add_argument("--teacher-forcing", default=0.5, type=float)
    parser.add_argument("--clip", default=1.0, type=float)
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="learning rate for optimizer"
    )
    parser.add_argument("--checkpoint", type=int, help="save model every N batches")
    parser.add_argument(
        "--skip-validate",
        default=False,
        action="store_true",
        help="set to False to skip validation",
    )
    parser.add_argument(
        "--freeze-embeddings",
        default=False,
        action="store_true",
        help="freeze source embedding layer",
    )
    parser.add_argument(
        "--continue-training-model",
        default=None,
        type=str,
        help="model for restarting training from a saved checkpoint",
    )
    main(parser.parse_args())
