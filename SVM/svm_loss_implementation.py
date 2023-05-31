import numpy as np
from svm_loss import SVMloss

# image matrix x
# correct class y
# parameter matrix W

x = np.array([1, 2, 3])
y = 2 
W = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])  # Matriz de parâmetros

# create instance of SVMloss()

svm_loss = SVMloss()

# call L_i_vectorized(x,y,W)

loss_i = svm_loss.L_i_vectorized(x, y, W)
print("Loss_i:", loss_i)