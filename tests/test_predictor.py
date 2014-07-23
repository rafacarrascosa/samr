import os
from unittest import TestCase

from samr import corpus
from samr.predictor import PhraseSentimentPredictor
from samr.data import Datapoint


TESTDATA_PATH = os.path.join(os.path.dirname(__file__), "data")


class TestPhraseSentimentPredictor(TestCase):
    def setUp(self):
        self.__original_path = corpus.DATA_PATH
        corpus.DATA_PATH = TESTDATA_PATH

    def tearDown(self):
        corpus.DATA_PATH = self.__original_path

    def test_fit_returns_self(self):
        train, _ = corpus.make_train_test_split("defiant order")
        predictor = PhraseSentimentPredictor()
        s = predictor.fit(train)
        self.assertEqual(predictor, s)

    def test_simple_predict(self):
        train, test = corpus.make_train_test_split("inhaler")
        predictor = PhraseSentimentPredictor()
        predictor.fit(train)
        predictions = predictor.predict(test)

        # Same amount of predictions than input values
        self.assertEqual(len(predictions), len(test))

        # Predicted labels where seen during training
        train_labels = set(x.sentiment for x in train)
        predicted_labels = set(predictions)
        self.assertEqual(predicted_labels - train_labels, set())

    def test_simple_error_matrix(self):
        train, test = corpus.make_train_test_split("reflektor", proportion=0.4)
        predictor = PhraseSentimentPredictor()
        predictor.fit(train)
        error = predictor.error_matrix(test)
        for real, predicted in error.keys():
            self.assertNotEqual(real, predicted)

        score = predictor.score(test)
        assert score > 0, "Test is valid only if score is more than 0"
        N = float(len(test))
        wrong = sum(len(xs) for xs in error.values())
        self.assertEqual((N - wrong) / N, score)

    def test_simple_duplicates(self):
        dupe = Datapoint(phraseid="a", sentenceid="b", phrase="b a", sentiment="1")
        # Train has a lot of "2" sentiments
        train = [Datapoint(phraseid=str(i),
                           sentenceid=str(i),
                           phrase="a b",
                           sentiment="2") for i in range(10)]
        train.append(dupe)
        test = [Datapoint(*dupe)]
        predictor = PhraseSentimentPredictor(duplicates=True)
        predictor.fit(train)
        predicted = predictor.predict(test)[0]
        self.assertEqual(predicted, "1")
