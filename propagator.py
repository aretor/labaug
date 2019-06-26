import os
import os.path as osp
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn import svm

import gtg


def factory(name, hard_labels):
    if name == 'gtg':
        return GTG(hard_labels)
    elif name == 'svm':
        return SVM(hard_labels)
    elif name == 'labels_only':
        return LabelsOnly(hard_labels)
    else:
        raise ValueError('algorithm \'' + name + '\' not recognized.')


class Propagator(object):
    def __init__(self, hard_labels):
        self.hard_labels = hard_labels
        self.alg_labels = None

    def _create_relabeled_file(self, fnames, new_file, labels, sep=' ', replace_labels=False, sep_replace=None):
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

    def save_labeling(self, fnames, alg_path):
        if not osp.exists(osp.dirname(alg_path)):
            os.makedirs(osp.dirname(alg_path))

        if (self.hard_labels and (self.alg_labels == -1).sum() > 0) or \
                (not self.hard_labels and (self.alg_labels.sum(axis=1) == 0.).sum() > 0):
            raise ValueError('there is some unlabeled observation, check the algorithm implementation,')

        self._create_relabeled_file(fnames, alg_path, self.alg_labels, sep=',')

    def propagate(self, features, alg_labels, labels, labeled, unlabeled, **kwargs):
        raise NotImplementedError('method \'propagate\' is abstract.')


class GTG(Propagator):
    def __init__(self, hard_labels):
        super().__init__(hard_labels)
        self.W = None
        self.features = None

    def _init_rand_probability(self, labels, labeled, unlabeled):
        nr_classes = int(labels.max() + 1)
        one_hots = np.zeros((labels.shape[0], nr_classes))
        one_hots[labeled, labels[labeled].ravel().astype(int)] = 1.0
        one_hots[unlabeled, :] = np.full((1, nr_classes), 1.0 / nr_classes)
        return one_hots

    def propagate(self, features, alg_labels, labels, labeled, unlabeled, **kwargs):
        """predict labels with gtg"""
        tol = kwargs.pop('tol', 1e-6)
        max_iter = kwargs.pop('max_iter', 25)

        if not np.array_equal(self.features, features):
            self.W = gtg.sim_mat(features, verbose=True)
            self.features = features

        ps = self._init_rand_probability(labels, labeled, unlabeled)
        res = gtg.gtg(W, ps, tol=tol, max_iter=max_iter, labels=labels, U=unlabeled, L=labeled)

        if self.hard_labels:
            self.alg_labels[unlabeled] = res[unlabeled].argmax(axis=1)
        else:
            self.alg_labels[unlabeled] = res[unlabeled]


class SVM(Propagator):
    def propagate(self, features, alg_labels, labels, labeled, unlabeled, **kwargs):
        """ predict labels with a linear SVM """
        lin_svm = svm.LinearSVC()

        if self.hard_labels:
            lin_svm.fit(features[labeled, :], labels[labeled])
            svm_labels = lin_svm.predict(features[unlabeled]).astype(int)
        else:
            cv = min(np.unique(labels[labeled], return_counts=True)[1].min(), 3)
            clf = CalibratedClassifierCV(lin_svm, cv=cv)
            clf.fit(features[labeled, :], labels[labeled])

            svm_labels = clf.predict_proba(features[unlabeled])

        self.alg_labels[unlabeled] = svm_labels


class LabelsOnly(Propagator):
    def save_labeling(self, fnames, alg_path):
        if not self.hard_labels:
            self.alg_labels = np.argmax(self.alg_labels, axis=1)

        fnames = [fnames[i] for i in self.alg_labels]
        super().save_labeling(fnames, alg_path)

    def propagate(self, features, alg_labels, labels, labeled, unlabeled, **kwargs):
        """ generate labeled only file """
        self.alg_labels = self.alg_labels[labeled]
