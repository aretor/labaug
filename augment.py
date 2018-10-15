import os
import os.path as osp
import pickle
import gtg
import augment_tools
from sklearn import svm

import config as cfg


def main2():
    dset_name = 'caltech'
    # open the file we have to fill
    results = 'results-' + dset_name + '.txt'
    with open(results, 'w') as file:
        file.write("Net_name,trial_ind,gtg,svm,ann\n")

    dset = cfg.DSETS[dset_name]
    data_dir = osp.join(cfg.DATA_DIR, dset_name)
    feat_dir = osp.join(data_dir, cfg.FEATURE_DIR)
    max_iter = 25
    tr_percs = [0.02, 0.05, 0.1]
    net_models = ['densenet121', 'resnet18']
    path_fun = lambda method, net_name, tr_perc, exp: osp.join(data_dir, method, net_name, 'labels_' + str(tr_perc) +
                                                                   '_' + str(exp) + '.txt')

    for exp in range(cfg.NUM_EXPS):
        for net_model in net_models:
            with open(osp.join(feat_dir, 'train', net_model + '_' + str(exp) + '.pickle'), 'r') as pkl:
                net_name, labels, features, fnames = pickle.load(pkl)
            W = gtg.sim_mat(features, verbose=True)

            for tr_perc in tr_percs:
                print('exp: ' + str(exp), net_model + ' - ' + str(dset['nr_classes']) + ' classes')
                labeled, unlabeled = augment_tools.create_equiclass_mapping(labels, tr_perc)
                ps = augment_tools.gen_init_rand_probability(labels, labeled, unlabeled)
                ps_new = gtg.gtg(W, ps, max_iter=max_iter, labels=labels, U=unlabeled, L=labeled)
                gtg_labels = ps_new.argmax(axis=1)
                # print('final accuracy: ', utils2.get_accuracy(ps_new, labels, unlabelled))

                # generate gtg label file
                gtg_path = path_fun('gtg', net_name, tr_perc, exp)
                if not osp.exists(osp.dirname(gtg_path)):
                    os.makedirs(osp.dirname(gtg_path))
                augment_tools.generate_relabeled_file(fnames, gtg_path, gtg_labels, sep=',')

                # do the same thing but with a linear SVM
                svm_linear_classifier = svm.LinearSVC()
                svm_linear_classifier.fit(features[labeled, :], labels[labeled])
                svm_labels = svm_linear_classifier.predict(features[unlabeled]).astype(int)
                gtg_labels[unlabeled] = svm_labels

                svm_path = path_fun('svm', net_name, tr_perc, exp)
                if not osp.exists(osp.dirname(svm_path)):
                    os.makedirs(osp.dirname(svm_path))
                augment_tools.generate_relabeled_file(fnames, svm_path, gtg_labels, sep=',')

                # generate labeled only file
                labeled_only_path = path_fun('labeled_only', net_name, tr_perc, exp)
                if not osp.exists(osp.dirname(labeled_only_path)):
                    os.makedirs(osp.dirname(labeled_only_path))
                labeled_only = [fnames[i] for i in labeled]
                augment_tools.generate_relabeled_file(labeled_only, labeled_only_path, gtg_labels[labels.astype(int)], sep=',')


if __name__ == '__main__':
    main2()
