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
    """
    Returns an iterable of `Datapoint`s with the contents of train.tsv.
    """
    if not __cached:
        __cached.extend(_iter_data_file("train.tsv"))
    return __cached


def iter_test_corpus():
    """
    Returns an iterable of `Datapoint`s with the contents of test.tsv.
    """
    return list(_iter_data_file("test.tsv"))


def make_train_test_split(seed, proportion=0.9):
    """
    Makes a randomized train/test split of the train.tsv corpus with
    `proportion` fraction of the elements going to train and the rest to test.
    The `seed` argument controls a shuffling of the corpus prior to splitting.
    The same seed should always return the same train/test split and different
    seeds should always provide different train/test splits.

    Return value is a (train, test) tuple where train and test are lists of
    `Datapoint` instances.
    """
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
