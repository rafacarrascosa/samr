"""
Generate a tsv submission file to kaggle's 'Sentiment Analysis on Movie Reviews' (samr)
competition using the samr module with a given json configuration file.
"""

if __name__ == "__main__":
    import argparse
    import json
    import csv
    import sys

    from samr.corpus import iter_corpus, iter_test_corpus
    from samr.predictor import PhraseSentimentPredictor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename")
    config = parser.parse_args()
    config = json.load(open(config.filename))

    predictor = PhraseSentimentPredictor(**config)
    predictor.fit(list(iter_corpus()))
    test = list(iter_test_corpus())
    prediction = predictor.predict(test)

    writer = csv.writer(sys.stdout)
    writer.writerow(("PhraseId", "Sentiment"))
    for datapoint, sentiment in zip(test, prediction):
        writer.writerow((datapoint.phraseid, sentiment))
