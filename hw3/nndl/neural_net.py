import numpy as np
import matplotlib.pyplot as plt

"""
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.
"""

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (H, D)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (C, H)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(hidden_size, input_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(output_size, hidden_size)
    self.params['b2'] = np.zeros(output_size)


  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Calculate the output scores of the neural network.  The result
    #   should be (C, N). As stated in the description for this class,
    #   there should not be a ReLU layer after the second FC layer.
    #   The output of the second FC layer is the output scores. Do not
    #   use a for loop in your implementation.
    # ================================================================ #

    # input - fully connected layer - ReLU - fully connected layer -> SCORES -> softmax
    #
    # 1st layer: h_1 = f(W_1*x_1 + b_1)  --> can let f(x) = x --> h_1 = W_1*x_1 + b_1
    # 2nd layer: h_2 = W_2*x_2 + b_2     --> output layer doesn't have a f()
    # f() is the activation function for when nonlinearity is introduced
    # wrap the layer output in it
    # in this case, our activation function = ReLU = max(0,x) --> np.maximum(0,x)
    # input -> h_1 = W_1*x_1 + b_1 -> f(h_1) -> W_2*x_2 + b_2 -> scores
    # at second layer, is x_2 == f(h_1) ?

    h1 = X.dot(W1.T) + b1
    relu = np.maximum(0,h1)
    scores = relu.dot(W2.T) + b2

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #


    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Calculate the loss of the neural network.  This includes the
    #   softmax loss and the L2 regularization for W1 and W2. Store the
    #   total loss in the variable loss.  Multiply the regularization
    #   loss by 0.5 (in addition to the factor reg).
    # ================================================================ #

    # scores is num_examples by num_classes

    # softmax loss --> exp(score) / sum{1...c} exp(score)
    # L2 regularization for W1 and W2 --> 0.5 * reg * W^2
    # you just add the L2 reg to the softmax? lol
    # loss = softmax + L2_reg

    # softmax numerator = score[right label] (1-D vector)
    # divide by sum of all scores for that label

    num_examples, num_classes = scores.shape
    softmax = np.zeros((num_examples, 1))

    for i in np.arange(num_examples):
      softmax[i] = np.exp(scores[i, y[i]]) / np.sum(np.exp(scores[i]))

    w1_reg = 0.5 * reg * np.sum(W1 ** 2)
    w2_reg = 0.5 * reg * np.sum(W2 ** 2)

    loss = -np.sum(np.log(softmax)) / num_examples
    loss = loss + w1_reg + w2_reg

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    grads = {}

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass.  Compute the derivatives of the
    #   weights and the biases.  Store the results in the grads
    #   dictionary.  e.g., grads['W1'] should store the gradient for
    #   W1, and be of the same size as W1.
    # ================================================================ #

    # X -> h1 -a-> relu -b-> h2 -c-> softmax -f-> L

    # dL/dc = dc/df * dL/df
    # dL/df = 1
    # dc/df = d(softmax)/d(h2)

    grad_softmax = np.zeros_like(scores)
    p = np.zeros_like(scores)

    scores -= np.max(scores, axis=1, keepdims=True)
    p = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    p[np.arange(num_examples), y] -= 1
    grad_softmax = p / num_examples

    grad_h2 = grad_softmax # dL/dc

    # dL/db = dc/db * dL/dc
    # dc/db = d(W2 * relu + b2)/d(relu)
    grad_relu = grad_h2.dot(W2)

    # dL/da = db/da * dL/db
    # db/da = 1 if h1 > 0, 0 if h1 < 0
    grad_h1 = (h1 > 0) * grad_relu

    # dL/dW2 = dc/dW2 * dL/dc
    # dc/dW2 = h1
    grads['W2'] = grad_h2.T.dot(relu) + reg * W2

    # dL/db2 = dc/db2 * dL/dc
    # dc/db2 = 1 * dL/dc
    grads['b2'] = np.sum(grad_h2, axis=0)

    # dL/dW1 = da/dW1 * dL/da
    # da/dW1 = X
    grads['W1'] = grad_h1.T.dot(X) + reg * W1

    # dL/db1 = da/db1 * dL/da
    # da/db1 = 1
    grads['b1'] = np.sum(grad_h1, axis=0)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in np.arange(num_iters):
      X_batch = None
      y_batch = None

      # ================================================================ #
      # YOUR CODE HERE:
      #   Create a minibatch by sampling batch_size samples randomly.
      # ================================================================ #
      idx = np.random.choice(num_train, batch_size)
      X_batch = X[idx]
      y_batch = y[idx]

      # ================================================================ #
      # END YOUR CODE HERE
      # ================================================================ #

       # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      # ================================================================ #
      # YOUR CODE HERE:
      #   Perform a gradient descent step using the minibatch to update
      #   all parameters (i.e., W1, W2, b1, and b2).
      # ================================================================ #

      self.params['W1'] -= learning_rate * grads['W1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['b2'] -= learning_rate * grads['b2']

      # ================================================================ #
      # END YOUR CODE HERE
      # ================================================================ #

      if verbose and it % 100 == 0:
        print('iteration {} / {}: loss {}'.format(it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Predict the class given the input data.
    # ================================================================ #
    h1 = X.dot(self.params['W1'].T) + self.params['b1']
    relu = np.maximum(0, h1)
    h2 = relu.dot(self.params['W2'].T) + self.params['b2']
    pred = np.exp(h2) / np.sum(np.exp(h2), axis=0)
    # print(X.shape)
    # print(pred.shape)
    y_pred = np.argmax(pred, axis=1)
    # print(y_pred.shape)


    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return y_pred


