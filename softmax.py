from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    n_sample = X.shape[0]
    n_class = W.shape[1]
    scores = X.dot(W)

    for i in range(n_sample):
        s_i = scores[i]
        softmax = np.exp(s_i) / np.sum(np.exp(s_i))
        # softmax loss
        loss += -np.log(softmax[y[i]])
        # gradient
        for j in range(n_class):
            dW[:, j] += X[i] * softmax[j]
        dW[:, y[i]] -= X[i] # 정답 레이블

    loss /= n_sample
    dW /= n_sample

    loss += reg * np.sum(W*W)
    dW += reg * 2 * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    n_samples = X.shape[0]
    n_class = W.shape[1]
    scores = X.dot(W)

    softmax = np.exp(scores) / np.sum(np.exp(scores),axis=1).reshape(-1,1)
    loss = -np.sum(np.log(softmax[range(n_samples,y)]))

    # Weight Gradient
    softmax[np.arange(n_samples), y] -= 1
    dW = X.T.dot(softmax)

    # Average
    loss /= n_samples
    dW /= n_samples

    # Regularization
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
