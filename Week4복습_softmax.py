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

    num_classese = W.shape[1]
    num_train = X.shape[0]

    for i in xrange(num_trian):
      scores = X[i].dot(W)
      scores -= max(scores)

      #loss
      loss_i = -score[y[i]] + np.log(sum(np.exp(scores)))
      loss += loss_i

      #grad
      for j in xrange(num_classes):
        softmax_output = np.exp(scores[j])/sum(np.exp(scores))
        dW[:,j] += softmax_output * X[i]

      dW[:, y[i]] -= X[i]
    
    loss/= num_train
    dW /= num_train

    loss += 0.5 * reg * np.sum(W*W)
    dW += 2 * reg * dW
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

    num_classes = W.shape[1]
    num_train = X.shape[0]  
    scores = X.dot(W)

    scores -= np.max(scores, axis = 1).reshape(-1,1)
    softmax_output = np.exp(scores)/np.sum(np.exp(scores), axis=1).reshape(-1,1)
    loss = -np.sum(np.log(softmax_output[range(num_train), y]))

    dS = softmax_output.copy()
    dS[range(num_train), list(y)] -= 1
    dW = (X.T).dot(dS)

    loss/= num_train
    dW /= num_train

    loss += 0.5 * reg * np.sum(W*W)
    dW += 2 * reg * dW

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
