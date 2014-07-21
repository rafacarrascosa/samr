"""
Run a 10-fold cross validation evaluation of a samr model given by a json
configuration file.
"""
import time


def fix_json_dict(config):
    new = {}
    for key, value in config.items():
        if isinstance(value, dict):
            value = fix_json_dict(value)
        elif isinstance(value, str):
            if value == "true":
                value = True
            elif value == "false":
                value = False
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
        new[key] = value
    return new


class PrintPartialCV:
    def __init__(self):
        self.last = time.time()
        self.i = 0

    def report(self, score):
        new = time.time()
        self.i += 1
        print("individual {}-th fold score={}% took {} seconds".format(self.i, score * 100, new - self.last))
        self.last = new


if __name__ == "__main__":
    import argparse
    import json

    from samr.evaluation import cross_validation
    from samr.predictor import PhraseSentimentPredictor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename")
    config = parser.parse_args()
    config = json.load(open(config.filename))

    factory = lambda: PhraseSentimentPredictor(**config)
    factory()  # Run once to check config is ok

    report = PrintPartialCV()
    result = cross_validation(factory, seed="robot rock", callback=report.report)

    print("10-fold cross validation score {}%".format(result * 100))
