import numpy as np

class NearestNeighbor(object):
    def __init__(self) -> None:
        pass

    def train(self,X,y,dist):
        """X is N x D where each row is an example. y is 1-dimension of size N.
        dist is for distance metrics {"l1", "l2"}."""
        self.Xtrain = X
        self.Ytrain = y
        self.dist = dist
    
    def predict(self,X):
        """X is N x D where each row is an example. We want to predict the labels."""
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.Ytrain.dtype)

        if self.dist == 'l1':
            for i in range(num_test):
                dist = np.sum(np.abs(self.Xtrain - X[i,:]),axis=1)
                min_index = np.argmin(dist)
                Ypred[i] = self.Ytrain[min_index]
        elif self.dist == 'l2':
            for i in range(num_test):
                dist = np.sqrt(np.sum(np.square(self.Xtrain - X[i,:]), axis = 1))
                min_index = np.argmin(dist)
                Ypred[i] = self.Ytrain[min_index]
        return Ypred