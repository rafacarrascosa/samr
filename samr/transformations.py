import re


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
