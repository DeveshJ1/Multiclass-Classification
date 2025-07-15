import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from skeleton_code import OneVsAllClassifier

class TestOneVsAllClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_classification(
            n_samples=100, n_features=20, n_informative=15, n_redundant=5,
            n_classes=3, random_state=42
        )
        cls.base_estimator = SVC(kernel='linear', probability=True)
        cls.n_classes = 3
        cls.classifier = OneVsAllClassifier(estimator=cls.base_estimator, n_classes=cls.n_classes)

    def test_fit(self):
        self.classifier.fit(self.X, self.y)
        self.assertTrue(self.classifier.fitted)
        for estimator in self.classifier.estimators:
            self.assertTrue(hasattr(estimator, "classes_"))

    def test_decision_function(self):
        self.classifier.fit(self.X, self.y)
        decision_scores = self.classifier.decision_function(self.X)
        self.assertEqual(decision_scores.shape, (self.X.shape[0], self.n_classes))

    def test_predict(self):
        self.classifier.fit(self.X, self.y)
        predictions = self.classifier.predict(self.X)
        self.assertEqual(predictions.shape, (self.X.shape[0],))
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions < self.n_classes))

    def test_accuracy(self):
        self.classifier.fit(self.X, self.y)
        predictions = self.classifier.predict(self.X)
        accuracy = accuracy_score(self.y, predictions)
        self.assertGreater(accuracy, 0.6)

if __name__ == '__main__':
    unittest.main()
