import os
from unittest import TestCase

from samr import corpus


TESTDATA_PATH = os.path.join(os.path.dirname(__file__), "data")


class TestCorpus(TestCase):
    def setUp(self):
        self.__original_path = corpus.DATA_PATH
        corpus.DATA_PATH = TESTDATA_PATH

    def tearDown(self):
        corpus.DATA_PATH = self.__original_path

    def test_make_train_test_split_simple(self):
        train, test = corpus.make_train_test_split("blitz")
        self.assertIn("word play", [x.phrase for x in train + test])
        self.assertEqual(len(set(x.sentenceid for x in test)), 1)
        self.assertEqual(len(set(x.sentenceid for x in test + train)), 4)

    def test_make_train_test_split_seed_works(self):
        a1, a2 = corpus.make_train_test_split("a")
        b1, b2 = corpus.make_train_test_split("b")
        c1, c2 = corpus.make_train_test_split("a")
        self.assertEqual(a1, c1)
        self.assertEqual(a2, c2)
        self.assertNotEqual(a1, b1)
        self.assertNotEqual(a2, b2)

    def test_make_train_test_split_no_shared_sentences(self):
        """
        Test that train and test don't share sent ids.
        """
        train, test = corpus.make_train_test_split("semis")
        train_ids = set(x.sentenceid for x in train)
        test_ids = set(x.sentenceid for x in test)
        self.assertEqual(train_ids & test_ids, set())

    def test_iter_test_corpus_simple(self):
        test = list(corpus.iter_test_corpus())
        self.assertEqual(len(test), 4)
        self.assertEqual(set("1 99 100 123".split()), set(x.phraseid for x in test))
        self.assertEqual(set("4 7 8 9".split()), set(x.sentenceid for x in test))
        self.assertIn("yo mama so fat", [x.phrase for x in test])
        self.assertEqual(set([None]), set(x.sentiment for x in test))

    def test_iter_data_file_bad_header(self):
        with self.assertRaises(ValueError):
            list(corpus._iter_data_file("badheader.tsv"))
