import os
import csv
import linecache
import subprocess
import spacy
import re
from sklearn.model_selection import train_test_split
from spacy.lang.ru import Russian
import argparse


def clean_and_tok(line, tokenizer, lower):

    # put a space between punctuation that comes after more than one letter (not abbreviations)
    line = re.sub('([.,!?":;])', r" \1", line)
    line = re.sub(r"(?<!\s)\(", r" (", line)
    line = re.sub(r"\(", r"( ", line)
    line = re.sub(r"\)", r" )", line)
    line = re.sub("https : //|http : //", "http://", line)
    # put a space between apostrophes that come before more than two letters (discludes 've and 'nt)
    line = re.sub(r"\'(?=[\-A-Za-z0-9]{3})", "' ", line)
    line = re.sub(r"\"(?=[\-A-Za-z0-9])", '" ', line)
    line = re.sub("”|“", '"', line)
    line = re.sub("@ ", "@", line)
    line = re.sub("# ", "#", line)
    line = re.sub("amp;", "&", line)
    line = re.sub(r"\*|¿|¡|^|<|>|~|\+|\=", "", line)
    line = re.sub("’|‘", "'", line)

    if lower == True:
        line = line.casefold()

    # optionally tokenize sentence
    if tokenizer is not None:
        line = tokenizer(line)
    else:
        line = line.split()

    return line, len(line)


def limit_length(src_list, trg_list, max_len=None, min_len=None):

    good_len_src = []
    good_len_trg = []

    for s, t in zip(src_list, trg_list):
        len_s, len_t = s[1], t[1]
        if (
            len_s >= min_len
            and len_s <= max_len
            and len_t >= min_len
            and len_t <= max_len
        ):
            good_len_src.append(s[0])
            good_len_trg.append(t[0])
    assert len(good_len_src) == len(good_len_trg)
    print(
        f"number of examples after removing sequences based on min and/or max length {len(good_len_src)}"
    )

    return good_len_src, good_len_trg


def prep_data_files(
    src_file,
    trg_file,
    save_path,
    src_tok,
    trg_tok,
    max_len=None,
    min_len=None,
    lower=False,
):
    X = []
    y = []
    # save data to temporary file
    with open(os.path.join(save_path, "temp_src.txt"), "w") as sink:
        total_length = int(
            subprocess.check_output("wc -l " + src_file, shell=True).split()[0]
        )
        for i in range(total_length):
            line = linecache.getline(src_file, i)
            line, len_line = clean_and_tok(line, src_tok, lower)
            if max_len or min_len:
                X.append((i, len_line))
            else:
                X.append(i)
            sink.write(" ".join(line))
    with open(os.path.join(save_path, "temp_trg.txt"), "w") as sink:
        total_length = int(
            subprocess.check_output("wc -l " + trg_file, shell=True).split()[0]
        )
        for i in range(total_length):
            line = linecache.getline(trg_file, i)
            line, len_line = clean_and_tok(line, trg_tok, lower)
            if max_len or min_len:
                y.append((i, len_line))
            else:
                y.append(i)
            sink.write(" ".join(line))
    assert len(X) == len(y)
    print(f"Total number of examples {len(X)}")
    if max_len or min_len:
        X, y = limit_length(X, y, max_len, min_len)

    return X, y


def split_to_tsv(split, X, y, save_path):
    fields = ["src", "trg"]
    src = os.path.join(save_path, "temp_src.txt")
    trg = os.path.join(save_path, "temp_trg.txt")
    source = [linecache.getline(src, i).strip() for i in X]
    target = [linecache.getline(trg, i).strip() for i in y]
    with open(os.path.join(save_path, f"{split}.tsv"), "w") as sink:
        csv_writer = csv.writer(sink, delimiter="\t")
        csv_writer.writerow(fields)
        csv_writer.writerows(zip(source, target))


def main(args):

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    tokenizers = {
        "en": spacy.load("en_core_web_sm"),
        "zh": spacy.load("zh_core_web_sm"),
        "ru": Russian(),
        "fr": spacy.load("fr_core_news_sm"),
        "es": spacy.load("es_core_news_sm"),
    }

    if args.src_tok is not None:
        spacy_src = tokenizers[args.src_tok]

        def tokenize_src(text):
            return [tok.text for tok in spacy_src.tokenizer(text)]

        src_tokenizer = tokenize_src
    else:
        src_tokenizer = None

    if args.trg_tok is not None:
        spacy_trg = tokenizers[args.trg_tok]

        def tokenize_trg(text):
            return [tok.text for tok in spacy_trg.tokenizer(text)]

        trg_tokenizer = tokenize_trg
    else:
        trg_tokenizer = None

    X, y = prep_data_files(
        args.src_file,
        args.trg_file,
        args.save_path,
        src_tok=src_tokenizer,
        trg_tok=trg_tokenizer,
        max_len=55,
        min_len=5,
        lower=True,
    )

    X_train, X, y_train, y = train_test_split(X, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # free up some memory by deleting unnecessary variables
    del (X, y)

    split_to_tsv("train", X_train, y_train, args.save_path)
    split_to_tsv("test", X_test, y_test, args.save_path)
    split_to_tsv("valid", X_valid, y_valid, args.save_path)

    # delete temporary files
    os.remove(os.path.join(args.save_path, "temp_src.txt"))
    os.remove(os.path.join(args.save_path, "temp_trg.txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-file", type=str, help="filename of source language file")
    parser.add_argument("--trg-file", type=str, help="filename of source language file")
    parser.add_argument("--save-path", help="folder for saving train,test,valid files")
    parser.add_argument(
        "--max-len", default=None, type=int, help="maximum sequence length"
    )
    parser.add_argument(
        "--min-len", default=None, type=int, help="minimum sequence length"
    )
    parser.add_argument(
        "--src-tok",
        default=None,
        type=str,
        choices=["en", "es", "fr", "ru", "zh"],
        help="source language tokenizer. options are en, es, fr, ru, or zh",
    )
    parser.add_argument(
        "--trg-tok",
        default=None,
        type=str,
        choices=["en", "es", "fr", "ru", "zh"],
        help="target language tokenizer. options are en, es, fr, ru, or zh",
    )
    main(parser.parse_args())
