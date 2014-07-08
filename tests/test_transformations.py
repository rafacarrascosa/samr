from unittest import TestCase

from samr.transformations import ReplaceText


class TestReplaceText(TestCase):
    def test_empty(self):
        r = ReplaceText([])
        Z = r.transform([])
        self.assertEqual(len(Z), 0)
        X = ["Deadmau5 4x4 = 12"]
        r = ReplaceText([])
        Z = r.transform(X)
        self.assertEqual(list(Z), X)

    def test_fit_returns_self(self):
        r = ReplaceText([])
        s = r.fit([])
        self.assertEqual(s, r)

    def test_simple(self):
        X = ["Sentence number one number two and so on .",
             "Old ubuntu version is 12.04, but it's still mantained"]
        Y = ["Sentence number one number two and so on ",
             "Old ubuntu version is 1204, but it is still mantained"]
        r = ReplaceText([
            (".", ""),
            ("'s", " is"),
        ])
        Z = r.transform(X)
        self.assertEqual(Z, Y)

    def test_priority_is_accounted(self):
        X = ["What ' is ' what should n't be and what ' will be '"]
        Y = ["What  is  what should not be and what  will be "]
        r = ReplaceText([
            ("n't", "not"),
            ("'", ""),
        ])
        Z = r.transform(X)
        self.assertEqual(Z, Y)
