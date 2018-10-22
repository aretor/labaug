import os
import os.path as osp
import pickle
import numpy as np
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV


import gtg
from augment_tools import equiclass_mapping, init_rand_probability, \
    create_relabeled_file


class Augmenter(object):
    def __init__(self, dset, splitting_dir, feat_dir, label_dir, net_names):
        self.dset = dset
        self.net_names = net_names
        self.splitting_dir = splitting_dir
        self.feat_dir = feat_dir
        self.label_dir = label_dir

    def __call__(self, *args, **kwargs):
        tr_percs = kwargs.pop('tr_percs', [0.02, 0.05, 0.1])
        algs = kwargs.pop('algs', ['gtg', 'svm', 'labels_only'])
        hard_labels = kwargs.pop('hard_labels', True)
        max_iter = kwargs.pop('max_iter', 25)

        if not osp.exists(self.label_dir):
            os.makedirs(self.label_dir)

        with open(osp.join(self.splitting_dir, 'test.txt'), 'r') as src,\
             open(osp.join(self.label_dir, 'test_labels.txt'), 'w') as dst:
            for line in src:
                dst.write(osp.join(self.dset['src'], line.rstrip() +
                                   ',' + str(int(line[:3]) - 1)) + '\n')

        for net_name in self.net_names:
            with open(osp.join(self.feat_dir, 'train', net_name + '.pickle'),
                      'r') as pkl:
                net_name, labels, features, fnames = pickle.load(pkl)
                labels = labels.ravel()

                # uncomment to fa debug code
                # labels = labels[:5000]
                # features = features[:5000]
                # fnames = fnames[:5000]

            for tr_perc in tr_percs:
                labeled, unlabeled = equiclass_mapping(labels, tr_perc)
                for alg in algs:
                    print(net_name + ' - ' + str(self.dset['nr_classes']) +
                          ' classes')

                    # generate alg label file name
                    alg_path = osp.join(self.label_dir, alg, net_name,
                                        'labels_{}.txt'.format(tr_perc))

                    if hard_labels:
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

                        if hard_labels:
                            alg_labels[unlabeled] = res[unlabeled].argmax(axis=1)
                        else:
                            alg_labels[unlabeled] = res[unlabeled]

                    elif alg == 'svm':
                        # predict labels with a linear SVM
                        lin_svm = svm.LinearSVC()

                        if hard_labels:
                            lin_svm.fit(features[labeled, :], labels[labeled])
                            svm_labels = lin_svm.predict(features[unlabeled]).astype(int)
                        else:
                            clf = CalibratedClassifierCV(lin_svm)
                            clf.fit(features[labeled, :], labels[labeled])
                            svm_labels = clf.predict_proba(
                                features[unlabeled])

                        alg_labels[unlabeled] = svm_labels

                    elif alg == 'labels_only':
                        # generate labeled only file
                        alg_labels = alg_labels[labeled]

                        if not osp.exists(osp.dirname(alg_path)):
                            os.makedirs(osp.dirname(alg_path))

                        if (hard_labels and (alg_labels == -1).sum() > 0) or \
                                (not hard_labels and (alg_labels.sum(
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

                    if (hard_labels and (alg_labels == -1).sum() > 0) or\
                        (not hard_labels and (alg_labels.sum(axis=1) == 0.).sum() > 0):
                        raise ValueError('There is some unlabeled observation,'
                                         'check \'' + alg + '\' algorithm,')

                    create_relabeled_file(fnames, alg_path, alg_labels, sep=',')

            if 'W' in locals():
                del W