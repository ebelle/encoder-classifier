import os
import csv
import torch
from torchtext.data import Field, TabularDataset
from torchtext.vocab import Vectors
import argparse
from collections import Counter
import linecache
import numpy as np


def load_train_data(SRC, TRG, data_path, source, target):
    data_fields = [(source, SRC), (target, TRG)]
    train_data = TabularDataset(
        path=os.path.join(data_path, "train.tsv"),
        format="tsv",
        fields=data_fields,
        skip_header=True,
    )
    return train_data


def load_all_data(SRC, TRG, data_path, source, target):
    # sometimes all the tags are not in the training set
    # so we load the full dataset
    data_fields = [(source, SRC), (target, TRG)]
    train, valid, test = TabularDataset.splits(
        path=data_path,
        train="train.tsv",
        validation="valid.tsv",
        test="test.tsv",
        format="tsv",
        fields=data_fields,
        skip_header=True,
    )
    return train, valid, test


# TODO: add in target vectors option
def main(args):

    if args.task == "translation":

        SRC = Field(init_token="<sos>", eos_token="<eos>", include_lengths=True)
        TRG = Field(init_token="<sos>", eos_token="<eos>")

        train_data = load_train_data(
            SRC, TRG, args.data_path, args.source_name, args.target_name
        )

        if args.source_vectors is not None:
            SRC_VECTORS = Vectors(name=args.source_vectors)
            SRC.build_vocab(
                train_data, max_size=args.max_vocab_size, vectors=SRC_VECTORS
            )
        else:
            SRC.build_vocab(train_data, max_size=args.max_vocab_size)

        torch.save(SRC, os.path.join(args.data_path, "src_vocab.pt"))
        print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
        TRG.build_vocab(train_data, max_size=args.max_vocab_size)
        torch.save(TRG, os.path.join(args.data_path, "trg_vocab.pt"))
        print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")

    elif args.task == "tagging":
        SRC = Field()
        TRG = Field(unk_token=None)

        train, valid, test = load_all_data(
            SRC, TRG, args.data_path, args.source_name, args.target_name
        )

        # uses source vocab from translation task so we only build target vocab
        TRG.build_vocab(train, valid, test)
        torch.save(TRG, os.path.join(args.data_path, "trg_vocab.pt"))
        print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")
        # clear space
        del train, valid, test, TRG

        if args.baseline:
            c = Counter()
            lens = []
            for name in ["train.tsv", "valid.tsv", "test.tsv"]:
                filename = os.path.join(args.data_path, name)
                total_length = sum(1 for _ in open(filename, "r"))
                # -1 in the length and +2 in the getline skips the header
                for i in range(total_length - 1):
                    line, tags = linecache.getline(filename, i + 2).split(
                        "\t"
                    )  # [1].split()
                    tags = tags.split()
                    lens.append(len(tags))
                    c.update(tags)
            total = sum(c.values())
            largest = c.most_common(1)
            percent_of = largest[0][1] / total
            print()
            print(
                f"Mean sentence length: {np.mean(lens):.2f} | Min: {min(lens)} | Max: {max(lens)}"
            )
            print()
            print(
                f"Total tokens: {total:,} | Largest class: {largest[0][0]}  {largest[0][1]:,} | Percent of total: {percent_of:.2f}"
            )
            print()
            print(f"All counts: \n {c}")


if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", help="folder where data is stored")
    parser.add_argument(
        "--task", choices=["translation", "tagging"], help="translation or tagging",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        help="title of source column in tsv. If translation_data_prep was used, it will be 'src'",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        help="title of target column in tsv. If translation_data_prep was used, it will be 'src'",
    )
    parser.add_argument("--max-vocab-size", default=None, type=int)
    parser.add_argument(
        "--source-vectors",
        default=None,
        help="optionally add word embeddings for source",
    )
    parser.add_argument(
        "--baseline",
        default=False,
        action="store_true",
        help="Do not get counts and baseline for tags",
    )

    main(parser.parse_args())
