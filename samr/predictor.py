from collections import defaultdict

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import accuracy_score

from samr.transformations import (ExtractText, ReplaceText, MapToSynsets,
                                  Densifier, ClassifierOvOAsFeatures)
from samr.inquirer_lex_transform import InquirerLexTransform


_valid_classifiers = {
    "sgd": SGDClassifier,
    "knn": KNeighborsClassifier,
    "svc": SVC,
    "randomforest": RandomForestClassifier,
}


def target(phrases):
    return [datapoint.sentiment for datapoint in phrases]


class PhraseSentimentPredictor:
    def __init__(self, classifier="sgd", classifier_args=None, lowercase=True,
                 text_replacements=None, map_to_synsets=False, binary=False,
                 min_df=0, ngram=1, stopwords=None, limit_train=None,
                 map_to_lex=False, duplicates=False):
        self.limit_train = limit_train
        self.duplicates = duplicates

        # Build pre-processing common to every extraction
        pipeline = [ExtractText(lowercase)]
        if text_replacements:
            pipeline.append(ReplaceText(text_replacements))

        # Build feature extraction schemes
        ext = [build_text_extraction(binary=binary, min_df=min_df,
                                     ngram=ngram, stopwords=stopwords)]
        if map_to_synsets:
            ext.append(build_synset_extraction(binary=binary, min_df=min_df,
                                               ngram=ngram))
        if map_to_lex:
            ext.append(build_lex_extraction(binary=binary, min_df=min_df,
                                            ngram=ngram))
        ext = make_union(*ext)
        pipeline.append(ext)

        # Build classifier and put everything togheter
        if classifier_args is None:
            classifier_args = {}
        classifier = _valid_classifiers[classifier](**classifier_args)
        self.pipeline = make_pipeline(*pipeline)
        self.classifier = classifier

    def fit(self, phrases, y=None):
        y = target(phrases)
        if self.duplicates:
            self.dupes = DuplicatesHandler()
            self.dupes.fit(phrases, y)
        Z = self.pipeline.fit_transform(phrases, y)
        if self.limit_train:
            self.classifier.fit(Z[:self.limit_train], y[:self.limit_train])
        else:
            self.classifier.fit(Z, y)
        return self

    def predict(self, phrases):
        Z = self.pipeline.transform(phrases)
        labels = self.classifier.predict(Z)
        if self.duplicates:
            for i, phrase in enumerate(phrases):
                label = self.dupes.get(phrase)
                if label is not None:
                    labels[i] = label
        return labels

    def score(self, phrases):
        pred = self.predict(phrases)
        return accuracy_score(target(phrases), pred)

    def error_matrix(self, phrases):
        predictions = self.predict(phrases)
        matrix = defaultdict(list)
        for phrase, predicted in zip(phrases, predictions):
            if phrase.sentiment != predicted:
                matrix[(phrase.sentiment, predicted)].append(phrase)
        return matrix


def build_text_extraction(binary, min_df, ngram, stopwords):
    return make_pipeline(CountVectorizer(binary=binary,
                                         tokenizer=lambda x: x.split(),
                                         min_df=min_df,
                                         ngram_range=(1, ngram),
                                         stop_words=stopwords),
                         ClassifierOvOAsFeatures())


def build_synset_extraction(binary, min_df, ngram):
    return make_pipeline(MapToSynsets(),
                         CountVectorizer(binary=binary,
                                         tokenizer=lambda x: x.split(),
                                         min_df=min_df,
                                         ngram_range=(1, ngram)),
                         ClassifierOvOAsFeatures())


def build_lex_extraction(binary, min_df, ngram):
    return make_pipeline(InquirerLexTransform(),
                         CountVectorizer(binary=binary,
                                         tokenizer=lambda x: x.split(),
                                         min_df=min_df,
                                         ngram_range=(1, ngram)),
                         Densifier())


class DuplicatesHandler:
    def fit(self, phrases, target):
        self.dupes = {}
        for phrase, label in zip(phrases, target):
            self.dupes[self._key(phrase)] = label

    def get(self, phrase):
        key = self._key(phrase)
        return self.dupes.get(key)

    def _key(self, x):
        return " ".join(x.phrase.lower().split())


class _Baseline:
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return ["2" for _ in X]

    def score(self, X):
        gold = target(X)
        pred = self.predict(X)
        return accuracy_score(gold, pred)
