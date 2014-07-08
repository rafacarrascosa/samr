from samr.corpus import make_train_test_split


def cross_validation(factory, seed, K=10):
    seed = str(seed)
    scores = []
    for k in range(K):
        train, test = make_train_test_split(seed + str(k))
        predictor = factory()
        predictor.fit(train)
        scores.append(predictor.score(test))
    return sum(scores) / len(scores)
