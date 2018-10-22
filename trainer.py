import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from extract_tools import get_finetune_model, prepare_loader
import net


class Trainer(object):
    def __init__(self, dset, label_dir, net_dir, res_dir, net_names):
        self.dset = dset
        self.net_names = net_names
        self.label_dir = label_dir
        self.net_dir = net_dir
        self.res_dir = res_dir

    def __call__(self, *args, **kwargs):
        tr_percs = kwargs.pop('tr_percs', [0.02, 0.05, 0.1])
        algs = kwargs.pop('algs', ['gtg', 'svm', 'labels_only'])
        batch_size = kwargs.pop('batch_size_tr', 8)
        epochs = kwargs.pop('epochs', 10)
        lr = kwargs.pop('lr', 1e-4)
        hard_labels = kwargs.pop('hard_labels', True)

        if not osp.exists(self.net_dir):
            os.makedirs(self.net_dir)

        ts_lab_path = osp.join(self.label_dir, 'test_labels.txt')
        for net_name in self.net_names:
            res_model_dir = osp.join(self.res_dir, net_name)
            if not osp.exists(res_model_dir):
                os.makedirs(res_model_dir)

            for tr_perc in tr_percs:
                for alg in algs:
                    lab_path = osp.join(self.label_dir, alg, net_name,
                                        'labels_{}.txt'.format(tr_perc))

                    train_loader = prepare_loader(lab_path, '', self.dset['stats'],
                                                  batch_size, shuffle=True,
                                                  sep=',', hard_labels=hard_labels)

                    test_loader = prepare_loader(ts_lab_path, '', self.dset['stats'],
                                                 batch_size, shuffle=False,
                                                 sep=',', hard_labels=hard_labels)

                    model = get_finetune_model(net_name, self.dset['nr_classes'])
                    criterion = nn.CrossEntropyLoss() if hard_labels else\
                        nn.KLDivLoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    trained_net = net.train(model, net_name, train_loader,
                                            test_loader, optimizer, criterion,
                                            epochs, self.net_dir, hard_labels=hard_labels)

                    model.load_state_dict(torch.load(trained_net))
                    accuracy, P, R, F1, conf_mat = net.evaluate(model,
                                                                test_loader)

                    with open(osp.join(res_model_dir,
                                       'res_{}_{}.txt'.format(alg, tr_perc)),
                              'w') as res:
                        res.write('\nACC,P,R,F1\n')
                        res.write(
                            '{},{},{},{}\n'.format(accuracy, P, R, F1))

                    np.savetxt(osp.join(res_model_dir,
                                        'conf_mat_{}_{}.txt'.format(alg,
                                                                    tr_perc)),
                               conf_mat)
