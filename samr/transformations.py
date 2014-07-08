class ExtractText:
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [datapoint.phrase for datapoint in X]
