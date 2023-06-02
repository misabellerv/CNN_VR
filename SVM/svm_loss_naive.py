import numpy as np

def L_i_vectorized(x, y, W):
    """
    Compute multiclass SVM loss for an example.
    Inputs:
    x: np.array (image) (Nx1)
    W: np.array (parameter matrix) (num_class x N)
    y: int (correct label)
    Outputs:
    loss_i: float (loss for that image)
    """
    scores = W.dot(x)
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i

def L(X, y, W):
    """
  fully-vectorized implementation:
  Not considering bias!
  - X holds all the training examples as columns (DxN)
  - y is array of integers specifying correct class (Nx1)
  - W are weights (KxD)
    """
    N = len(y)
    scores = X @ W  
    y_hat_true = scores[range(N), y][:, np.newaxis]
    margins = np.maximum(0, scores - y_hat_true + 1)
    loss = margins.sum() / N 
    return loss