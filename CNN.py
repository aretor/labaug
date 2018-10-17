import os
import os.path as osp
import torch
import torch.nn as nn
import torch.optim as optim

import config as cfg
from extract_tools import prepare_loader, get_finetune_model
import net

dset_name = 'caltech'
dataset = cfg.DSETS[dset_name]
model_names = ['resnet18', 'densenet121']
data_dir = osp.join(cfg.DATA_DIR, dset_name)
feature_dir = osp.join(data_dir, cfg.FEATURE_DIR)
label_dir = osp.join(data_dir, cfg.LABEL_DIR)
net_dir = osp.join(data_dir, cfg.NET_DIR)
epochs = 10
batch_size = 8
tr_percs = [0.02, 0.05, 0.1]

if not osp.exists(net_dir):
    os.makedirs(net_dir)

for exp in range(cfg.NUM_EXPS):
    ts_lab_path = osp.join(label_dir, 'test_labels_{}.txt'.format(exp))
    for model_name in model_names:
        for tr_perc in tr_percs:
            for alg in ['gtg', 'svm', 'labeled_only']:
                lab_path = osp.join(label_dir, alg, model_name,
                                    'labels_{}_{}.txt'.format(tr_perc, exp))

                train_loader = prepare_loader(lab_path, '', dataset['stats'],
                                              batch_size, shuffle=True, sep=',')

                test_loader = prepare_loader(ts_lab_path, '', dataset['stats'],
                                             batch_size, shuffle=False, sep=',')

                model = get_finetune_model(model_name, dataset['nr_classes'])
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=1e-4)

                trained_net = net.train(model, model_name, train_loader,
                                        test_loader, optimizer, criterion,
                                        epochs, net_dir, exp)

                model.load_state_dict(torch.load(trained_net))
                net_accuracy = net.evaluate(model, test_loader)
                print('Final accuracy: {}'.format(net_accuracy))

# # do the same thing but with a linear SVM
# svm_linear_classifier = svm.LinearSVC()
# svm_linear_classifier.fit(features[labeled, :], labels[labeled])
# labels_svm = svm_linear_classifier.predict(features[unlabeled])

# labels_svm = labels_svm.astype(int)
# gtg_labels[unlabeled] = labels_svm
#
# svm_label_file = osp.join(svm_labels_dir, model_name + '.txt')
# utils2.gen_gtg_label_file(fnames, names_folds, gtg_labels, svm_label_file)
# gen_gtg_dataset('indoors/train_' + str(ind), svm_label_file, ind, 'train_labeled_svm')
#
# dataset_train = os.path.join(dataset, 'train_labeled_svm_' + ind)
#
# train_loader = prepare_loader_train(dataset_train, stats, batch_size)
# test_loader = prepare_loader_val(dataset_test, stats, batch_size)
#
# net, feature_size = create_net(nr_classes, nets_and_features, net_type=nname)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=1e-4)
#
# trained_net = net.train(net, nname, train_loader, test_loader, optimizer, criterion, epochs, net_dir, ind)
#
# net.load_state_dict(torch.load(trained_net))
# net_accuracy_svm = evaluate(net, test_loader)
# print('Accuracy: ' + str(net_accuracy_svm))

# # now check the accuracy of the net trained only in the labeled set
# label_file = osp.join(only_labeled, nname + '.txt')
# utils2.only_labeled_file(fnames, labeled, label_file)
# gen_labeled_dataset('indoors/train_' + str(ind), label_file, ind)
#
# dataset_train = os.path.join(dataset, 'train_only_labeled_' + ind)
#
# train_loader = prepare_loader_train(dataset_train, stats, batch_size)
# test_loader = prepare_loader_val(dataset_test, stats, batch_size)
#
# net, feature_size = create_net(nr_classes, nets_and_features, net_type=nname)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=1e-4)
#
# trained_net = train(net, nname, train_loader, test_loader, optimizer, criterion, epochs, nets_dir_test, ind)
#
# net.load_state_dict(torch.load(trained_net))
# net_accuracy = evaluate(net, test_loader)

# # # finally, do gtg with the testing set
# # with open(os.path.join(feature_test_dir, pkl_name), 'rb') as pkl:k
# #     net_name_test, labels_test, features_test, fnames_test = pickle.load(pkl)
# #
# # features_combined = np.vstack((features[labeled,:], features_test))
# # labels_combined = np.vstack((labels[labeled], labels_test))
# # W = gtg.sim_mat(features_combined)
# # labeled = np.arange(features[labeled,:].shape[0])
# # unlabeled = np.arange(features[labeled,:].shape[0], features_combined.shape[0])
# #
# # ps = utils2.gen_init_rand_probability(labels_combined, labeled, unlabeled, nr_classes)
# # gtg_accuracy_test, Ps_new = utils2.get_accuracy(W, ps, labels_combined, labeled, unlabeled, len(unlabeled))
#
# with open(results, 'a') as file:
#     file.write(
#         nname + "   " + ind + "   " + str(net_accuracy_gtg) + "   " + str(net_accuracy_svm) + "   " + str(
#             net_accuracy) + "\n")

# print()
