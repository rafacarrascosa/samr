"""
SAMR main module, PhraseSentimentPredictor is the class that does the
prediction and therefore one of the main entry points to the library.
"""
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
    """
    Main `samr` class. It implements a trainable predictor for phrase
    sentiments. API is a-la scikit-learn, where:
        - `__init__` configures the predictor
        - `fit` trains the predictor from data. After calling `fit` the instance
          methods should be free side-effect.
        - `predict` generates sentiment predictions.
        - `score` evaluates classification accuracy from a test set.

    Outline of the predictor pipeline is as follows:
    A configurable main classifier is trained with a concatenation of 3 kinds of
    features:
        - The decision functions of set of vanilla SGDClassifiers trained in a
          one-versus-others scheme using bag-of-words as features.
        - (Optionally) The decision functions of set of vanilla SGDClassifiers
          trained in a one-versus-others scheme using bag-of-words on the
          wordnet synsets of the words in a phrase.
        - (Optionally) The amount of "positive" and "negative" words in a phrase
          as dictated by the Harvard Inquirer sentiment lexicon

    Optionally, during prediction, it also checks for exact duplicates between
    the training set and the train set.    """
    def __init__(self, classifier="sgd", classifier_args=None, lowercase=True,
                 text_replacements=None, map_to_synsets=False, binary=False,
                 min_df=0, ngram=1, stopwords=None, limit_train=None,
                 map_to_lex=False, duplicates=False):
        """
        Parameter description:
            - `classifier`: The type of classifier used as main classifier,
              valid values are "sgd", "knn", "svc", "randomforest".
            - `classifier_args`: A dict to be passed as arguments to the main
              classifier.
            - `lowercase`: wheter or not all words are lowercased at the start of
              the pipeline.
            - `text_replacements`: A list of tuples `(from, to)` specifying
              string replacements to be made at the start of the pipeline (after
              lowercasing).
            - `map_to_synsets`: Whether or not to use the Wordnet synsets
              feature set.
            - `binary`: Whether or not to count words in the bag-of-words
              representation as 0 or 1.
            - `min_df`: Minumim frequency a word needs to have to be included
              in the bag-of-word representation.
            - `ngram`: The maximum size of ngrams to be considered in the
              bag-of-words representation.
            - `stopwords`: A list of words to filter out of the bag-of-words
              representation. Can also be the string "english", in which case
              a default list of english stopwords will be used.
            - `limit_train`: The maximum amount of training samples to give to
              the main classifier. This can be useful for some slow main
              classifiers (ex: svc) that converge with less samples to an
              optimum.
            - `max_to_lex`: Whether or not to use the Harvard Inquirer lexicon
              features.
            - `duplicates`: Whether or not to check for identical phrases between
              train and prediction.
        """
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
        """
        `phrases` should be a list of `Datapoint` instances.
        `y` should be a list of `str` instances representing the sentiments to
        be learnt.
        """
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
        """
        `phrases` should be a list of `Datapoint` instances.
        Return value is a list of `str` instances with the predicted sentiments.
        """
        Z = self.pipeline.transform(phrases)
        labels = self.classifier.predict(Z)
        if self.duplicates:
            for i, phrase in enumerate(phrases):
                label = self.dupes.get(phrase)
                if label is not None:
                    labels[i] = label
        return labels

    def score(self, phrases):
        """
        `phrases` should be a list of `Datapoint` instances.
        Return value is a `float` with the classification accuracy of the
        input.
        """
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
