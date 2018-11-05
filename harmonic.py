import numpy as np


def harmonic_function(W, fl, L, U):
  l = len(L)  # the number of labeled points
  n = W.shape[0]  # totale nummber of points

  # the graph Laplacian L=D-W
  lap = np.diag(W.sum(axis=0)) - W

  # the harmonic function.
  fu = np.dot(np.dot(-np.linalg.inv(lap[np.ix_(U, U)]), lap[np.ix_(U, L)]), fl)

  # compute the CMN solution
  q = fl.sum(axis=0) + 1.  # the unnormalized class proportion estimate from labeled data, with Laplace smoothing
  fu_CMN = fu * np.tile(q / fu.sum(axis=0), (n - l, 1))

  return fu, np.argmax(fu, axis=1)
