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
        raise ValueError("Input file has wrong column names: {}".format(path))
    for row in it:
        if len(row) == 3:
            row += (None,)
        yield Datapoint(*row)


def iter_corpus(__cached=[]):
    if not __cached:
        __cached.extend(_iter_data_file("train.tsv"))
    return __cached


def iter_test_corpus():
    return list(_iter_data_file("test.tsv"))


def make_train_test_split(seed, proportion=0.9):
    data = list(iter_corpus())
    ids = list(sorted(set(x.sentenceid for x in data)))
    if len(ids) < 2:
        raise ValueError("Corpus too small to split")
    N = int(len(ids) * proportion)
    if N == 0:
        N += 1
    rng = random.Random(seed)
    rng.shuffle(ids)
    test_ids = set(ids[N:])
    train = []
    test = []
    for x in data:
        if x.sentenceid in test_ids:
            test.append(x)
        else:
            train.append(x)
    return train, test
