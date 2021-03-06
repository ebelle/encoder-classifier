import os
import csv
import linecache
import subprocess
import spacy
import re
from sklearn.model_selection import train_test_split
from spacy.lang.ru import Russian
from cltk.tokenize.word import WordTokenizer
import argparse
from flair.models import SequenceTagger
from flair.data import Sentence, Token
import time
from utils import epoch_time


def clean_and_tok(line, tokenizer):
    line = line.strip()
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

    # optionally tokenize sentence
    if tokenizer is not None:
        line = tokenizer(line)
    else:
        line = line.split()

    return line, len(line)


def limit_length(src_list, trg_list=None, max_len=None, min_len=None):

    good_len_sentences = []

    if trg_list:
        for s, t in zip(src_list, trg_list):
            len_s, len_t = s[1], t[1]
            if (
                len_s >= min_len
                and len_s <= max_len
                and len_t >= min_len
                and len_t <= max_len
            ):
                good_len_sentences.append(s[0])
        print(
            f"number of examples after removing sequences based on min and/or max length {len(good_len_sentences)}"
        )
    else:
        for s in src_list:
            len_s = s[1]
            if len_s >= min_len and len_s <= max_len:
                good_len_sentences.append(s[0])
        print(
            f"number of examples after removing sequences based on min and/or max length {len(good_len_sentences)}"
        )

    return good_len_sentences


def get_tags(line, tagger):
    # join list for tagging
    sentence = Sentence()
    for token in line:
        sentence.add_token(Token(token))
    tagger.predict(sentence)
    # split to get tags
    tagged_line = sentence.to_tagged_string().split()
    tags = []
    # tags are every other token in sentence
    for i in range(1, len(tagged_line), 2):
        tags.append(tagged_line[i][1:-1])
    return tags


def prep_trans_files(
    src_file, trg_file, save_path, src_tok, trg_tok, max_len, min_len,
):
    keep_indices = []
    X = []
    y = []
    # save data to temporary file
    with open(os.path.join(save_path, "temp_src.txt"), "w") as sink:
        total_length = sum(1 for _ in open(src_file, "r"))
        for i in range(total_length):
            line = linecache.getline(src_file, i)
            line, len_line = clean_and_tok(line, src_tok)
            if max_len or min_len:
                X.append((i, len_line))
            else:
                # keep all the data
                keep_indices.append((i, len_line))
            sink.write(" ".join(line) + "\n")
    with open(os.path.join(save_path, "temp_trg.txt"), "w") as sink:
        total_length = sum(1 for _ in open(trg_file, "r"))
        for i in range(total_length):
            line = linecache.getline(trg_file, i)
            line, len_line = clean_and_tok(line, trg_tok)
            if max_len or min_len:
                y.append((i, len_line))
            sink.write(" ".join(line) + "\n")
    assert len(X) == len(y)
    print(f"Total number of examples {len(X)}")
    if max_len or min_len:
        keep_indices = limit_length(X, y, max_len, min_len)

    return keep_indices


def prep_tag_files(
    src_file, save_path, src_tok, max_len, min_len,
):
    tagger = SequenceTagger.load("pos-fast")
    good_len_sentences = 0
    # save data to temporary file
    with open(os.path.join(save_path, "temp_src.txt"), "w") as src_sink:
        with open(os.path.join(save_path, "temp_trg.txt"), "w") as trg_sink:
            total_length = sum(1 for _ in open(src_file, "r"))
            print(f"total number of lines: {total_length}")
            start_time = time.time()
            for i in range(total_length):
                if i != 0 and i % 10000 == 0:
                    end_time = time.time()
                    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                    print(f"Line {i}| Time: {epoch_mins}m {epoch_secs}s")
                    start_time = time.time()
                line = linecache.getline(src_file, i + 1)
                line, len_line = clean_and_tok(line, src_tok)
                if max_len or min_len:
                    if len_line >= min_len and len_line <= max_len:
                        good_len_sentences += 1
                        src_sink.write(" ".join(line) + "\n")
                        trg_sink.write(" ".join(get_tags(line, tagger=tagger)) + "\n")

    keep_indices = [i for i in range(good_len_sentences)]
    print(f"Total number of examples {len(keep_indices)}")
    return keep_indices


def split_to_tsv(split, X, save_path):
    fields = ["src", "trg"]
    src = os.path.join(save_path, "temp_src.txt")
    trg = os.path.join(save_path, "temp_trg.txt")
    source = [linecache.getline(src, i + 1).strip() for i in X]
    target = [linecache.getline(trg, i + 1).strip() for i in X]
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
        "ar": WordTokenizer("arabic"),
    }

    src_tokenizer = None
    if args.src_tok is not None:
        src_tok = tokenizers[args.src_tok]
        if args.src_tok == "ar":

            def tokenize_src(text):
                return [tok for tok in src_tok.tokenize(text)]

        else:

            def tokenize_src(text):
                return [tok.text for tok in src_tok.tokenizer(text)]

        src_tokenizer = tokenize_src

    trg_tokenizer = None
    if args.trg_tok is not None:
        trg_tok = tokenizers[args.trg_tok]
        if args.trg_tok == "ar":

            def tokenize_trg(text):
                return [tok for tok in trg_tok.tokenize(text)]

        else:

            def tokenize_trg(text):
                return [tok.text for tok in tokz.tokenizer(text)]

        trg_tokenizer = tokenize_trg

    if args.task == "translation":
        indices = prep_trans_files(
            args.src_file,
            args.trg_file,
            args.save_path,
            src_tok=src_tokenizer,
            trg_tok=trg_tokenizer,
            max_len=args.max_len,
            min_len=args.min_len,
        )
    elif args.task == "tagging":
        indices = prep_tag_files(
            args.src_file,
            args.save_path,
            src_tok=src_tokenizer,
            max_len=args.max_len,
            min_len=args.min_len,
        )

    train, indices, = train_test_split(indices, test_size=0.3, random_state=42)
    valid, test = train_test_split(indices, test_size=0.5, random_state=42)

    split_to_tsv("train", train, args.save_path)
    split_to_tsv("test", test, args.save_path)
    split_to_tsv("valid", valid, args.save_path)

    # delete temporary files
    os.remove(os.path.join(args.save_path, "temp_src.txt"))
    os.remove(os.path.join(args.save_path, "temp_trg.txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        choices=["translation", "tagging"],
        help="tasks include translation and tagging",
    )
    parser.add_argument("--src-file", type=str, help="filename of source language file")
    parser.add_argument(
        "--trg-file", default=None, type=str, help="filename of source language file"
    )
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
        choices=["ar", "en", "es", "fr", "ru", "zh"],
        help="source language tokenizer. options are ar, en, es, fr, ru, or zh",
    )
    parser.add_argument(
        "--trg-tok",
        default=None,
        type=str,
        choices=["ar", "en", "es", "fr", "ru", "zh"],
        help="target language tokenizer. options are ar, en, es, fr, ru, or zh",
    )
    parser.add_argument(
        "--no-sort",
        default=True,
        action="store_true",
        help="do not sort the sentences by lengths.",
    )
    main(parser.parse_args())
