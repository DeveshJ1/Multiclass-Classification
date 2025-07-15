import unittest
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from skeleton_code import featureMap, sgd, MulticlassSVM

class TestMulticlassSVM(unittest.TestCase):

    def setUp(self):
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y = np.array([0, 1, 2, 0])
        self.num_classes = 3
        self.num_outFeatures = 6
        self.lam = 1.0

        self.Delta = lambda y_true, y_pred: 1 if y_true != y_pred else 0

    def test_featureMap(self):
        X_1D = np.array([1, 2, 3])
        X_2D = np.array([[1, 2], [3, 4]])

        output_1D = featureMap(X_1D, 1, self.num_classes)
        output_2D = featureMap(X_2D, 2, self.num_classes)

        self.assertEqual(output_1D.shape, (1, len(X_1D) * self.num_classes))
        self.assertEqual(output_2D.shape, (X_2D.shape[0], X_2D.shape[1] * self.num_classes))

    def test_sgd(self):
        def dummy_subgd(x, y, w):
            return np.ones_like(w)

        w = sgd(self.X, self.y, self.num_outFeatures, dummy_subgd, eta=0.1, T=10)
        self.assertEqual(w.shape[0], self.num_outFeatures)

    def test_MulticlassSVM_fit_and_predict(self):
        svm = MulticlassSVM(num_outFeatures=self.num_outFeatures, lam=self.lam, num_classes=self.num_classes, Delta=self.Delta)

        svm.fit(self.X, self.y, eta=0.1, T=10)
        self.assertTrue(svm.fitted)

        predictions = svm.predict(self.X)
        self.assertEqual(predictions.shape[0], self.X.shape[0])

    def test_decision_function(self):
        svm = MulticlassSVM(num_outFeatures=self.num_outFeatures, lam=self.lam, num_classes=self.num_classes, Delta=self.Delta)
        svm.fit(self.X, self.y, eta=0.1, T=10)

        scores = svm.decision_function(self.X)
        self.assertEqual(scores.shape, (self.X.shape[0], self.num_classes))

    def test_subgradient(self):
        svm = MulticlassSVM(num_outFeatures=self.num_outFeatures, lam=self.lam, num_classes=self.num_classes, Delta=self.Delta)
        
        x_sample = np.array([1, 2])
        y_sample = 1
        w_sample = np.random.rand(self.num_outFeatures)
        gradient = svm.subgradient(x_sample, y_sample, w_sample)

        # Check the gradient shape
        self.assertEqual(gradient.shape, w_sample.shape)

if __name__ == '__main__':
    unittest.main()
