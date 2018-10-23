import os
import os.path as osp
import random
import pickle
import numpy as np
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

import gtg


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


class Augmenter(object):
    """ Class to augment the labels of a dataset by propagating information of labeled observations to unlabeled ones
    """
    def __init__(self, dset, splitting_dir, feat_dir, label_dir, net_names, hard_labels):
        self.dset = dset
        self.net_names = net_names
        self.splitting_dir = splitting_dir
        self.feat_dir = feat_dir
        self.label_dir = label_dir
        self.hard_labels = hard_labels

    def __call__(self, *args, **kwargs):
        """ Augment the labels

            Inputs:
            tr_percs: percentage of splitting between labeled and unlabeled observations
            algs: methods to perform the label propagation
            max_iter: parameter for 'gtg': number of iterations
        """
        tr_percs = kwargs.pop('tr_percs', [0.02, 0.05, 0.1])
        algs = kwargs.pop('algs', ['gtg', 'svm', 'labels_only'])
        max_iter = kwargs.pop('max_iter', 25)

        if not osp.exists(self.label_dir):
            os.makedirs(self.label_dir)

        with open(osp.join(self.splitting_dir, 'test.txt'), 'r') as src,\
             open(osp.join(self.label_dir, 'test_labels.txt'), 'w') as dst:
            for line in src:
                dst.write(osp.join(self.dset['src'], line.rstrip() + ',' + str(int(line[:3]) - 1)) + '\n')

        for net_name in self.net_names:
            with open(osp.join(self.feat_dir, 'train', net_name + '.pickle'), 'r') as pkl:
                net_name, labels, features, fnames = pickle.load(pkl)
                labels = labels.ravel()

                # uncomment to debug code
                # labels = labels[:5000]
                # features = features[:5000]
                # fnames = fnames[:5000]

            for tr_perc in tr_percs:
                labeled, unlabeled = equiclass_mapping(labels, tr_perc)
                for alg in algs:
                    print(net_name + ' - ' + str(self.dset['nr_classes']) + ' classes')

                    # generate alg label file name
                    alg_path = osp.join(self.label_dir, alg, net_name,
                                        'labels_{}.txt'.format(tr_perc))

                    if self.hard_labels:
                        alg_labels = np.full(labels.shape[0], -1)
                        alg_labels[labeled] = labels[labeled]
                    else:
                        alg_labels = np.zeros((len(labels),
                                               self.dset['nr_classes']))
                        alg_labels[
                            labeled, labels[labeled].ravel().astype(int)] = 1.0

                    if alg == 'gtg':
                        # predict labels with gtg
                        if 'W' not in locals():
                            W = gtg.sim_mat(features, verbose=True)

                        ps = init_rand_probability(labels, labeled, unlabeled)
                        res = gtg.gtg(W, ps, max_iter=max_iter, labels=labels,
                                      U=unlabeled, L=labeled)

                        if self.hard_labels:
                            alg_labels[unlabeled] = res[unlabeled].argmax(axis=1)
                        else:
                            alg_labels[unlabeled] = res[unlabeled]

                    elif alg == 'svm':
                        # predict labels with a linear SVM
                        lin_svm = svm.LinearSVC()

                        if self.hard_labels:
                            lin_svm.fit(features[labeled, :], labels[labeled])
                            svm_labels = lin_svm.predict(features[unlabeled]).astype(int)
                        else:
                            cv = min(np.unique(labels[labeled],
                                               return_counts=True)[1].min(), 3)
                            clf = CalibratedClassifierCV(lin_svm, cv=cv)
                            clf.fit(features[labeled, :], labels[labeled])

                            svm_labels = clf.predict_proba(
                                features[unlabeled])

                        alg_labels[unlabeled] = svm_labels

                    elif alg == 'labels_only':
                        # generate labeled only file
                        alg_labels = alg_labels[labeled]

                        if not osp.exists(osp.dirname(alg_path)):
                            os.makedirs(osp.dirname(alg_path))

                        if (self.hard_labels and (alg_labels == -1).sum() > 0) or \
                                (not self.hard_labels and (alg_labels.sum(
                                    axis=1) == 0.).sum() > 0):
                            raise ValueError(
                                'There is some unlabeled observation,'
                                'check \'' + alg + '\' algorithm,')

                        create_relabeled_file([fnames[i] for i in labeled],
                                              alg_path, alg_labels, sep=',')
                        break
                    else:
                        raise ValueError('algorithm \'' + alg + '\' not recognized.')

                    if not osp.exists(osp.dirname(alg_path)):
                        os.makedirs(osp.dirname(alg_path))

                    if (self.hard_labels and (alg_labels == -1).sum() > 0) or\
                        (not self.hard_labels and (alg_labels.sum(axis=1) == 0.).sum() > 0):
                        raise ValueError('There is some unlabeled observation,'
                                         'check \'' + alg + '\' algorithm,')

                    create_relabeled_file(fnames, alg_path, alg_labels, sep=',')

            if 'W' in locals():
                del W
