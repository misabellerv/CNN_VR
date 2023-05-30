import numpy as np

class KNearestNeighbor(object):
    """ kNN classifier with l1 or l2 distance. User must choose."""
    
    def __init__(self,k,dist):
        self.k = k
        self.dist = dist
        
    def train(self,X,y):
        """
        The training memorizes the training data.
        X: numpy array of shape (n_train,D)
        y: numpy array of shape (n_train,)
        k: int, number of neighbors.
        num_loops: which implementation to use to compute distance
        between training and testing points.
        """
        self.X_train = X
        self.y_train = y
    
    def predict(self,X):
        """
        Predict labels of X_test.
        X: numpy array of shape (n_test,D)
        """
        if self.dist == "l1":
            dists = self.compute_l1_distances(X)
        elif self.dist == "l2":
            dists = self.compute_l2_distances(X)
        else:
            raise ValueError(f'Invalid distance. distance options are "l1" and "l2"')
        
        return self.predict_labels(dists,self.k)
    
    def compute_l1_distances(self,X):
        """
        Compute the l1 distance between each test point in X and each training point.

        """
        n_test = X.shape[0]
        n_train = self.X_train.shape[0]
        dists = np.zeros((n_test, n_train)) 
        dists = np.sum(np.abs(X[:, np.newaxis] - self.X_train), axis=2)

        return dists
    
    def compute_l2_distances(self,X):
        """
        Compute the l2 distance between each test point in X and each training point.

        """
        n_test = X.shape[0]
        n_train = self.X_train.shape[0]
        dists = np.zeros((n_test, n_train)) 
        dists = np.sqrt(
          -2 * (X @ self.X_train.T) +
          np.power(X, 2).sum(axis=1, keepdims=True) +
          np.power(self.X_train, 2).sum(axis=1, keepdims=True).T)

        return dists
    
    def predict_labels(self, dists, k):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
        gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
        test data, where y[i] is the predicted label for the test point X[i].  
        """
        n_test = dists.shape[0]
        y_pred = np.zeros(n_test)
        for i in range(n_test):
            closest_y = self.y_train[dists[i].argsort()[:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred