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

    s = np.dot(X,W)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            if y[i]==j:
                h = np.exp(s[i][j])/np.sum(np.exp(s[i]))
                loss_i = -np.log(h)
                loss += loss_i
                dS = (h-1)/s.shape[0]
                dW[:,j] += X[i]*dS

                for q in range(s.shape[1]):
                    if q==j:
                        continue
                    dS2 = np.exp(s[i][q])/np.sum(np.exp(s[i]))/s.shape[0]
                    dW[:,q] += X[i]*dS2
    loss = loss/s.shape[0]
    loss += reg*np.sum(W*W)
    dW += 2*W*reg

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

    s = np.dot(X,W)
    s_exp = np.exp(s)
    loss_i = s_exp[range(s.shape[0]),y]/np.sum(s_exp,axis=1)
    loss_i = -np.log(loss_i+0.0001)
    loss = np.mean(loss_i)
    loss += np.sum(W*W*reg)
    dS = s_exp/np.reshape(np.sum(s_exp,axis=1),[-1,1])
    dS[range(s.shape[0]),y] = s_exp[range(s.shape[0]),y]/np.sum(s_exp,axis=1) -1
    dS = dS/X.shape[0]
    dW = np.dot(X.T,dS)
    dW += 2*W*reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
