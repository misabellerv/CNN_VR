import numpy as np

# f(x,W) = Wx = scores of labels
# L_{i} = \sum_{j!=yi} max(0, s_{j}-s_{yi} + 1)
# Wx are the predicted scores
# scores[y] are the scores from correct labels

class SVMloss(object):
    """ SVM loss for an example of training set.
        L = 1/N sum{Li}
        We find Li."""

    def __init__(self):
        pass

    def L_i_vectorized(self, x, y, W):
        """
        Compute multiclass SVM loss.
        Inputs:
        x: np.array (image)
        W: np.array (parameter matrix)
        y: int (correct label)
        Outputs:
        loss_i: float (loss for that image)
        """
        self.x = x
        self.y = y
        self.W = W
        scores = W.dot(x)
        margins = np.maximum(0, scores - scores[y] + 1)
        margins[y] = 0
        loss_i = np.sum(margins)
        return loss_i