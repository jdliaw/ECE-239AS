import numpy as np
import pdb

"""
This code was based off of code from cs231n at Stanford University, and modified for ece239as at UCLA.
"""

class KNN(object):

  def __init__(self):
    pass

  def train(self, X, y):
    """
	Inputs:
	- X is a numpy array of size (num_examples, D)
	- y is a numpy array of size (num_examples, )
    """
    self.X_train = X
    self.y_train = y

  def compute_distances(self, X, norm=None):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.
	- norm: the function with which the norm is taken.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    if norm is None:
      norm = lambda x: np.sqrt(np.sum(x**2))
      #norm = 2

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in np.arange(num_test):
      for j in np.arange(num_train):
        # ================================================================ #
        # YOUR CODE HERE:
        #   Compute the distance between the ith test point and the jth
        #   training point using norm(), and store the result in dists[i, j].
        # ================================================================ #

        # Euclidean distance: dist = sqrt(sum_1_2((x_i - x_j)^2)
        # Want to pass norm x_i - x_j == arg 'x'

        dists[i, j] = norm(X[i] - self.X_train[j])
    return dists

  def compute_L2_distances_vectorized(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train WITHOUT using any for loops.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))

    # ================================================================ #
    # YOUR CODE HERE:
    #   Compute the L2 distance between the ith test point and the jth
    #   training point and store the result in dists[i, j].  You may
    #    NOT use a for loop (or list comprehension).  You may only use
    #     numpy operations.
    #
    #     HINT: use broadcasting.  If you have a shape (N,1) array and
    #   a shape (M,) array, adding them together produces a shape (N, M)
    #   array.
    # ================================================================ #

    # X shape = (num_test, num_features) = (5000, 3072)
    # X_train , num_train = 500

    # X^2 - 2*XY + Y^2

    # sum the norms for X and X_rtrain separately (each element, not the whole matrix)
    X_norm = np.array(np.sum(X**2, axis=1)) # (500,)
    Y_norm = np.array(np.sum(self.X_train**2, axis=1)) # (5000,)

    # add (5000,) and (500,) to get (5000,500) thru broadcasting
    X_sum = Y_norm.reshape((num_train, 1)) + X_norm
    dists = np.sqrt(X_sum.T - 2*X.dot(self.X_train.T)) # transpose to get (500,5000) for both matrices

    return dists


  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in np.arange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      # ================================================================ #
      # YOUR CODE HERE:
      #   Use the distances to calculate and then store the labels of
      #   the k-nearest neighbors to the ith test point.  The function
      #   numpy.argsort may be useful.
      #
      #   After doing this, find the most common label of the k-nearest
      #   neighbors.  Store the predicted label of the ith training example
      #   as y_pred[i].  Break ties by choosing the smaller label.
      # ================================================================ #

      # find k nearest neighbors to X[i]
      # labels = y_train
      dists_i = dists[i] # all distances of ith test to training points
      sorted_dists = list(np.argsort(dists_i))
      k_sorted_dists = sorted_dists[:k]

      label_freq = {}

      for j in np.arange(k):
        label = self.y_train[k_sorted_dists[j]]
        closest_y.append(label)
        label_freq[label] = label_freq.get(label, 0) + 1


      max_freq = 0
      max_label = None
      for key,v in label_freq.items():
        if v > max_freq:
          max_freq = v
          max_label = key
        elif v == max_freq:
          max_label = min(max_label, key)

      y_pred[i] = max_label

    return y_pred
