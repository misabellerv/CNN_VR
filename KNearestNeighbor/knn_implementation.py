import numpy as np
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
from knn import KNearestNeighbor
from tqdm import tqdm

# load CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# reshape 1-D data
X_train = X_train.reshape(X_train.shape[0], 32*32*3)
X_test = X_test.reshape(X_test.shape[0], 32*32*3)

# call KNN
k = 5
dist = 'l2'
knn = KNearestNeighbor(k=k, dist=dist)

# training
knn.train(X_train, y_train)

# prediction
y_pred = np.zeros_like(y_test)
with tqdm(total=len(X_test), desc="Predicting") as pbar:
    for i in range(len(X_test)):
        y_pred[i] = knn.predict(X_test[i:i+1])
        pbar.update(1)

# accuracy
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy score kNN with k={k}, dist={dist}: {acc} ')