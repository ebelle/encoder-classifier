import os
import csv
import torch
from torchtext.data import Field, TabularDataset
from torchtext.vocab import Vectors
import argparse


def load_train_data(SRC, TRG, data_path, source, target):
    data_fields = [(source, SRC), (target, TRG)]
    train_data = TabularDataset(
        path=os.path.join(data_path, "train.tsv"),
        format="tsv",
        fields=data_fields,
        skip_header=True,
    )
    return train_data


def load_data(SRC, TRG, data_path, source, target):
    data_fields = [(source, SRC), (target, TRG)]
    data = TabularDataset.splits(
        path=data_path,
        train="train.tsv",
        validation="valid.tsv",
        test="test.tsv",
        format="tsv",
        fields=data_fields,
        skip_header=True,
    )
    return data


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
        return SRC, TRG

    elif args.task == "classification":
        SRC = Field()
        TRG = Field(unk_token=None)

        train, valid, test = load_data(
            SRC, TRG, args.data_path, args.source_name, args.target_name
        )

        # uses source vocab from translation task so we only build target vocab
        TRG.build_vocab(train, valid, test)
        torch.save(TRG, os.path.join(args.data_path, "trg_vocab.pt"))
        print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")
        return TRG


if __name__ == "__main__":
    # hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", help="folder where data is stored")
    parser.add_argument(
        "--task",
        choices=["translation", "classification"],
        help="translation or classification",
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

    main(parser.parse_args())
