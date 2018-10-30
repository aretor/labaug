import numpy as np
import sklearn
from math import log
import random


def create_relabeled_file(fnames, new_file, labels, sep=' ',
                          replace_labels=False, sep_replace=None):
    """ Generate a file containing a labeling of a dataset. Each line of the
        contains a path to am object and its class provided as an hard or soft
        label.

        Input:
        fnames: iterable of the paths to the labeled object
        new_file: name of the labeling file that will be created
            labels: iterable containing the labels, which can be provided as
            integers (hard labels) or ndarrays (soft labels)
        replace_labels: if elements in fnames already contains a label and this
            option is set to true labels will be replaced with the newly provided
            ones.
        sep_replace: separator to be used to separate labels from paths in case
            replace_labels is set to True (the default is 'sep')

    """
    if len(fnames) != len(labels):
        raise ValueError('length of filenames differs from length of labels')

    if replace_labels and not sep_replace:
        sep_replace = sep

    if isinstance(fnames, file):
        fnames = list(fnames.open('r'))

    with open(new_file, 'w') as fw:
        for row, lab in zip(fnames, labels):
            if replace_labels:
                row = row.split(sep_replace)[0]

            if labels.ndim == 1:
                fw.write(row + sep + str(lab) + '\n')
            else:
                fw.write(row + sep)
                np.savetxt(fw, lab, newline=' ')
                fw.write('\n')

    if isinstance(fnames, file):
        fnames.close()


def one_hot(labels, nr_classes):
    if len(labels.shape) == 2:
        labels = labels[:, 0]
    labels = labels.astype(int)
    one_hot_labels = np.zeros((labels.size, nr_classes))
    one_hot_labels[np.arange(labels.size), labels] = 1.
    return one_hot_labels


def create_mapping(nr_objects, percentage_labels):
    mapping = np.arange(nr_objects)
    np.random.shuffle(mapping)
    nr_labelled = int(percentage_labels * nr_objects)
    labelled = mapping[:nr_labelled]
    unlabelled = mapping[nr_labelled:]
    return np.sort(labelled), np.sort(unlabelled)


def equiclass_mapping(labels, label_perc):
    nr_classes = int(labels.max() + 1)

    labeled, unlabeled = [], []
    for n_class in range(nr_classes):
        class_labels = list(np.where(labels == n_class)[0])
        split = int(label_perc * len(class_labels))
        random.shuffle(class_labels)
        labeled += class_labels[:split]
        unlabeled += class_labels[split:]
    return np.array(labeled), np.array(unlabeled)


def init_rand_probability(labels, labeled, unlabeled):
    nr_classes = int(labels.max() + 1)
    labels_one_hot = np.zeros((labels.shape[0], nr_classes))
    labels_one_hot[labeled, labels[labeled].ravel().astype(int)] = 1.0
    labels_one_hot[unlabeled, :] = np.full((1, nr_classes), 1.0 / nr_classes)
    return labels_one_hot


def gen_init_probability(W, labels, labelled, unlabelled):
    """
    :param W: similarity matrix to generate the labels for the unlabelled observations
    :param labels: labels of the already labelled observations
    :return:
    """
    n = W.shape[0]
    k = int(log(n) + 1.)
    # labelled, unlabelled = create_mapping(n, perc_lab)
    W = W[np.ix_(unlabelled, labelled)]

    ps = np.zeros(labels.shape)
    ps[labelled] = labels[labelled]

    max_k_inds = labelled[np.argpartition(W, -k, axis=1)[:, -k:]]
    tmp = np.zeros((unlabelled.shape[0], labels.shape[1]))
    for row in max_k_inds.T:
        tmp += labels[row]
    tmp /= float(k)
    ps[unlabelled] = tmp

    return ps


def get_accuracy(P, labels, unlabeled):
    conf = sklearn.metrics.confusion_matrix(labels[unlabeled, :], (P[unlabeled, :]).argmax(axis=1))
    return float(conf.trace()) / conf.sum()


def gen_gtg_label_file(fnames, names_folds, labels_GT, out_fname):
    with open(out_fname, 'w') as file:
        for i in range(len(fnames)):
            splitted_name = fnames[i][0].split('/')
            new_name = splitted_name[8] + '/' + splitted_name[9] + ' ' + names_folds[labels_GT[i]] + "\n"
            file.write(new_name)


# file = open('only_labelled.txt', 'w')
# # and here we create a similar file just for the labelled data
# for i in range(len(names_of_files)):
#     splitted_name = names_of_files[i][0].split('/')
#     if i in labelled:
#         new_name = splitted_name[8] + '/' + splitted_name[9] + ' ' + splitted_name[8] + "\n"
#         file.write(new_name)
# file.close()


def unit_test():
    """
    unit_test for gen_init_probability function
    :return:
    """
    np.random.seed(314)
    # unlab = 0, 1, 3. lab = 2, 4, 5
    W = np.array([[5, 3, (8), 4, (9), (1)],
                  [1, 2, (3), 4, (7), (9)],
                  [7, 1, 2, 8, 4, 3],
                  [9, 7, (4), 3, (2), (1)],
                  [5, 7, 4, 2, 8, 6],
                  [6, 4, 5, 3, 1, 2]])

    labels = np.array([[0, 1, 0, 0, 0],
                       [1, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 1],
                       [0, 1, 0, 0, 0]])

    res = np.array([[0, 0, 0.5, 0, 0.5],
                    [0, 0.5, 0, 0, 0.5],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0.5, 0, 0.5],
                    [0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0]])
    print(gen_init_probability(W, labels, 0.5))
