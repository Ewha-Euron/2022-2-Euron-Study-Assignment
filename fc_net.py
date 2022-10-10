from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params["W1"] = np.random.normal(0.0, weight_scale, (input_dim, hidden_dim))
        self.params["W2"] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
        self.params["b1"] = np.zeros(hidden_dim, dtype=float)
        self.params["b2"] = np.zeros(num_classes, dtype=float)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out1, cache1 = affine_relu_forward(X, self.params["W1"], self.params["b1"])
        out2, cache2 = affine_forward(out1, self.params["W2"], self.params["b2"])
        scores = out2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dscores = softmax_loss(scores, y)
        dout1, grads["W2"], grads["b2"] = affine_backward(dscores, cache2)
        dX, grads["W1"], grads["b1"] = affine_relu_backward(dout1, cache1)

        for w in ["W2", "W1"]:
            if self.reg > 0:
                loss += 0.5 * self.reg * (self.params[w] ** 2).sum()
            grads[w] += self.reg * self.params[w]


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

class FullyConnectedNet(object):

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):

        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
        self.params['b1'] = np.zeros(hidden_dims[0])
        if self.use_batchnorm:
            self.params['gamma1'] = np.ones(hidden_dims[0])
            self.params['beta1'] = np.zeros(hidden_dims[0])

        for i in range(2, self.num_layers):
            self.params['W'+repr(i)] = weight_scale * np.random.randn(hidden_dims[i-2], hidden_dims[i-1])
            self.params['b'+repr(i)] = np.zeros(hidden_dims[i-1])
            if self.use_batchnorm:
                self.params['gamma'+repr(i)] = np.ones(hidden_dims[i-1])
                self.params['beta'+repr(i)] = np.zeros(hidden_dims[i-1])

        self.params['W'+repr(self.num_layers)] = weight_scale * np.random.randn(hidden_dims[self.num_layers-2], num_classes)
        self.params['b'+repr(self.num_layers)] = np.zeros(num_classes)

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):

        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None

        cache = {}
        dropout_cache = {}
        prev_hidden = X

        for i in range(1, self.num_layers):
            if self.use_batchnorm:
                prev_hidden, cache[i] = affine_bn_relu_forward(prev_hidden, self.params['W'+repr(i)], self.params['b'+repr(i)], 
                                                               self.params['gamma'+repr(i)], self.params['beta'+repr(i)], 
                                                               self.bn_params[i-1])
            else:
                prev_hidden, cache[i] = affine_relu_forward(prev_hidden, self.params['W'+repr(i)], self.params['b'+repr(i)])
            if self.use_dropout:
                prev_hidden, dropout_cache[i] = dropout_forward(prev_hidden, self.dropout_param)

        scores, cache[self.num_layers] = affine_forward(prev_hidden, self.params['W'+repr(self.num_layers)],
                                                        self.params['b'+repr(self.num_layers)])

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        loss, dscores = softmax_loss(scores, y)
        sum_squares_of_w = 0
        for i in range(1, self.num_layers+1):
            sum_squares_of_w += np.sum(self.params['W'+repr(i)]*self.params['W'+repr(i)])
        loss += 0.5 * self.reg * sum_squares_of_w

        dhidden, dW, db = affine_backward(dscores, cache[self.num_layers])
        dW += self.reg * self.params['W'+repr(self.num_layers)]
        grads['W'+repr(self.num_layers)] = dW
        grads['b'+repr(self.num_layers)] = db

        for i in list(reversed(range(1, self.num_layers))):
            if self.use_dropout:
                dhidden = dropout_backward(dhidden, dropout_cache[i])
            if self.use_batchnorm:
                dhidden, dW, db, dgamma, dbeta = affine_bn_relu_backward(dhidden, cache[i])
                grads['gamma'+repr(i)] = dgamma
                grads['beta'+repr(i)] = dbeta
            else:
                dhidden, dW, db = affine_relu_backward(dhidden, cache[i])
            dW += self.reg * self.params['W'+repr(i)]
            grads['W'+repr(i)] = dW
            grads['b'+repr(i)] = db

        return loss, grads