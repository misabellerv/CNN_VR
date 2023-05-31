import numpy as np

class oNN(object):
    def __init__(self,dist='l2'):
        self.dist =dist

    def train(self,X,y):
        """X is N x D where each row is an example. y is 1-dimension of size N.
        dist is for distance metrics {"l1", "l2"}."""
        self.Xtrain = X
        self.Ytrain = y
    
    def predict(self,X):
        """X is N x D where each row is an example. We want to predict the labels."""
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.Ytrain.dtype)

        if self.dist == "l1":
            distances = np.sum(np.abs(self.Xtrain[:, np.newaxis] - X), axis=2)
        elif self.dist == "l2":
            distances = np.linalg.norm(self.Xtrain[:, np.newaxis] - X, axis=2)
        else:
            raise ValueError("Invalid distance metric.")

        min_indices = np.argmin(distances, axis=0)
        Ypred = self.Ytrain[min_indices]

        return Ypred