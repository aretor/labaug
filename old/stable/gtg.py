import numpy as np
import sklearn.metrics
import time
from scipy import spatial


def one_hot(labels, nr_classes):
    labels = labels.astype(int)
    one_hot_labels = np.zeros((labels.size, nr_classes))
    one_hot_labels[np.arange(labels.size), labels] = 1.
    return one_hot_labels


def get_accuracy(P_new, labels, unlabelled):
    """
    This function computes the accuracy in the testing set
    :param fc7_features: fc7 features for both training and testing set
    :param softmax_features: softmax features for both training and testing set
    :param labels: labels for both training and testing set
    :param accuracy_cnn: the accuracy of cnn (baseline)
    :param testing_set_size: the size of the testing set
    :return: accuracy of our method, accuracy of cnn
    """
    if len(labels.shape) == 1:
        labels = one_hot(labels, labels.max() + 1)
    conf = sklearn.metrics.confusion_matrix(labels[unlabelled, :].argmax(axis=1), (P_new[unlabelled, :]).argmax(axis=1))
    return float(conf.trace()) / float(conf.sum())

# def gtg(W, X, max_iter=100, labels=None, U=None):
#     iter = 0
#
#     while iter < max_iter:
#         tmp = X * np.dot(W, X)
#         X = tmp / tmp.sum(axis=1)[:, np.newaxis]
#         iter += 1
#         if labels is not None and U is not None:
#             # conf = sklearn.metrics.confusion_matrix(labels[U, :], (X[U, :]).argmax(axis=1))
#             conf = sklearn.metrics.confusion_matrix(labels[U], (X[U, :]).argmax(axis=1))
#             acc = float(conf.trace()) / conf.sum()
#             print('Accuracy at iter ' + str(iter) + ': ' + str(acc))
#
#     return X


def gtg(W, X, L, U, max_iter=100, labels=None):
    iter = 0
    Xbin = X[L, :] > 0.0

    while iter < max_iter:
        q = ((X * np.dot(W[:, U], X[U, :])) + (X * np.dot(W[:, L], Xbin))).sum(axis=1)
        u = (np.dot(W[:, U], X[U, :]) + (np.dot(W[:, L], Xbin)))/q[:, np.newaxis]
        # DO not do in-place multiplication!
        X = X * u

        # checking step-by-step accuracy
        if (not labels is None):
            # conf = sklearn.metrics.confusion_matrix(labels[U, :], (X[U, :]).argmax(axis=1))
            conf = sklearn.metrics.confusion_matrix(labels[U], (X[U, :]).argmax(axis=1))
            acc = float(conf.trace()) / conf.sum()
            print('Accuracy at iter ' + str(iter) + ': ' + str(acc))

        iter += 1

    return X


# def gtg(W, X, L, U, max_iter=10, labels=None):
#     iter = 0
#     Xbin = X[L, :] > 0.0
#     # if (not labels is None) and len(labels.shape) == 1:
#     #     labels = one_hot(labels, labels.max() + 1)
#
#     while iter < max_iter:
#         q = ((X * np.dot(W[:, U], X[U, :])) + (X * np.dot(W[:, L], Xbin))).sum(axis=1)
#         u = (np.dot(W[:, U], X[U, :]) + (np.dot(W[:, L], Xbin)))/q[:, np.newaxis]
#         # DO not do in-place multiplication!
#         X = X * u
#
#         # checking step-by-step accuracy
#         if (not labels is None):
#             conf = sklearn.metrics.confusion_matrix(labels[U, :], (X[U, :]).argmax(axis=1))
#             acc = float(conf.trace()) / conf.sum()
#             print('Accuracy at iter ' + str(iter) + ': ' + str(acc))
#
#         iter += 1
#
#     return X


def sim_mat(fc7_feats, mode=0, verbose=False):
    """
    Given a matrix of features, generate the similarity matrix S and sparsify it.
    :param fc7_feats: the fc7 features
    :return: matrix_S - the sparsified matrix S
    """
    t = time.time()
    pdist_ = spatial.distance.pdist(fc7_feats)
    if verbose:
        print('Created distance matrix' + ' ' + str(time.time() - t) + ' sec')

    t = time.time()
    dist_mat = spatial.distance.squareform(pdist_)
    if verbose:
        print('Created square distance matrix' + ' ' + str(time.time() - t) + ' sec')
    del pdist_

    t = time.time()
    if mode == 0:
        sigmas = np.sort(dist_mat, axis=1)[:, 7] + 1e-16
    else:
        sigmas = (np.sort(dist_mat, axis=1)[:, 1:8] + 1e-16).mean(axis=1)
    sigma_prods = np.dot(sigmas[:, np.newaxis], sigmas[np.newaxis, :])
    if verbose:
        print('Generated sigmas' + ' ' + str(time.time() - t) + ' sec')

    t = time.time()
    dist_mat /= -sigma_prods
    if verbose:
        print('Computed dists/-sigmas' + ' ' + str(time.time() - t) + ' sec')

    del sigma_prods

    t = time.time()
    W = np.exp(dist_mat, dist_mat)
    # W = np.exp(-(dist_mat / matrice_prodotti_sigma))
    np.fill_diagonal(W, 0.)

    # sparsify the matrix
    k = int(np.floor(np.log2(fc7_feats.shape[0])) + 1)
    n = W.shape[0]
    if verbose:
        print('Created inplace similarity matrix' + ' ' + str(time.time() - t) + ' sec')

    t = time.time()
    for x in W:
        x[np.argpartition(x, n - k)[:(n - k)]] = 0.0
    if verbose:
        print('Sparsify the matrix' + ' ' + str(time.time() - t) + ' sec')

    t = time.time()
    # matrix_S = np.zeros((n, n))
    m1 = W[np.triu_indices(n, k=1)]
    m2 = W.T[np.triu_indices(n, k=1)]

    W = spatial.distance.squareform(np.maximum(m1, m2))
    if verbose:
        print('Symmetrized the similarity matrix' + ' ' + str(time.time() - t) + ' sec')

    return W
