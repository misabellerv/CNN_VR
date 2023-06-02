import numpy as np
from svm_loss_naive import L

# Set a matrix of 3 images 
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [1, 1, 1]])

# y corresponds to the correct classes to each image
y = np.array([0,2,1])

# W must be (KxD) = (2x3)
W = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6],
              [0.1, 0.5, -0.7]])

# Class loss function
loss = L(X, y, W)

print(f'Total loss from 3 images: {loss}')