from collections import defaultdict

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from samr.transformations import ExtractText, ReplaceText


_valid_classifiers = {
    "sgd": SGDClassifier,
}


def target(phrases):
    return [datapoint.sentiment for datapoint in phrases]


class PhraseSentimentPredictor:
    def __init__(self, classifier="sgd", classifier_args=None, lowercase=True,
                 text_replacements=None):
        if classifier_args is None:
            classifier_args = {}

        pipeline = [("extractor", ExtractText())]
        if text_replacements:
            pipeline.append(("replacements", ReplaceText(text_replacements)))
        pipeline.append(("vectorizer", CountVectorizer(lowercase=lowercase)))
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
