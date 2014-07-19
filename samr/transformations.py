import re

import nltk


class ExtractText:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [datapoint.phrase for datapoint in X]


class ReplaceText:
    def __init__(self, replacements):
        """
        Replacements should be a `(from, to)` tuple of strings.
        """
        self.rdict = dict(replacements)
        self.pat = re.compile("|".join(re.escape(origin) for origin, _ in replacements))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.rdict:
            return X
        return [self.pat.sub(self._repl_fun, x) for x in X]

    def _repl_fun(self, match):
        return self.rdict[match.group()]


class MapToSynsets:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self._text_to_synsets(x) for x in X]

    def _text_to_synsets(self, text):
        result = []
        for word in text.split():
            ss = nltk.wordnet.wordnet.synsets(word)
            result.extend(str(s) for s in ss if ".n." not in str(s))
        return " ".join(result)
