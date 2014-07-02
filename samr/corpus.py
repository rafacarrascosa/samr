import os
import csv
import random

from samr.data import Datapoint
from samr.settings import DATA_PATH


def _iter_data_file(filename):
    path = os.path.join(DATA_PATH, filename)
    it = csv.reader(open(path, "r"), delimiter="\t")
    row = next(it)  # Drop column names
    if " ".join(row[:3]) != "PhraseId SentenceId Phrase":
        raise ValueError("Input file has wrong column names", row)
    for row in it:
        if len(row) == 3:
            row += (None,)
        if len(row) != 4:
            raise ValueError("Wrong amount of columns in csv: {}".format(path))
        yield Datapoint(*row)


def iter_corpus():
    yield from _iter_data_file("train.tsv")


def iter_test_corpus():
    yield from _iter_data_file("test.tsv")


def make_train_test_split(seed, proportion=0.9):
    data = list(iter_corpus())
    if len(data) < 2:
        raise ValueError("Corpus too small to split")
    N = int(len(data) * proportion)
    if N == 0:
        N += 1
    rng = random.Random(seed)
    rng.shuffle(data)
    return data[:N], data[N:]
