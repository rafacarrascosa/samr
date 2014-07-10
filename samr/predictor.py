from collections import defaultdict

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from samr.transformations import ExtractText, ReplaceText, MapToSynsets


_valid_classifiers = {
    "sgd": SGDClassifier,
    "knn": KNeighborsClassifier,
    "svc": SVC,
}


def target(phrases):
    return [datapoint.sentiment for datapoint in phrases]


class PhraseSentimentPredictor:
    def __init__(self, classifier="sgd", classifier_args=None, lowercase=True,
                 text_replacements=None, map_to_synsets=False, binary=False):
        if classifier_args is None:
            classifier_args = {}

        pipeline = [("extractor", ExtractText())]
        if text_replacements:
            pipeline.append(("replacements", ReplaceText(text_replacements)))
        if map_to_synsets:
            pipeline.append(("synsets", MapToSynsets()))
        pipeline.append(("vectorizer", CountVectorizer(lowercase=lowercase,
                                                       binary=binary,
                                                       tokenizer=lambda x: x.split())))
        pipeline.append(("classifier", _valid_classifiers[classifier](**classifier_args)))
        self.pipeline = Pipeline(pipeline)

    def fit(self, phrases, y=None):
        self.pipeline.fit(phrases, target(phrases))
        return self

    def predict(self, phrases):
        return self.pipeline.predict(phrases)

    def score(self, phrases):
        return self.pipeline.score(phrases, target(phrases))

    def error_matrix(self, phrases):
        predictions = self.predict(phrases)
        matrix = defaultdict(list)
        for phrase, predicted in zip(phrases, predictions):
            if phrase.sentiment != predicted:
                matrix[(phrase.sentiment, predicted)].append(phrase)
        return matrix


class _Baseline:
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return ["2" for _ in X]

    def score(self, X):
        gold = target(X)
        pred = self.predict(X)
        return accuracy_score(gold, pred)
