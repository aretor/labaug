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
model_names = ['resnet18',
               'densenet121'
               ]
data_dir = osp.join(cfg.DATA_DIR, dset_name)
feature_dir = osp.join(data_dir, cfg.FEATURE_DIR)
label_dir = osp.join(data_dir, cfg.LABEL_DIR)
net_dir = osp.join(data_dir, cfg.NET_DIR)
res_dir = osp.join(data_dir, cfg.RESULT_DIR)
epochs = 10
batch_size = 8
tr_percs = [0.02,
            0.05,
            0.1
            ]

if not osp.exists(net_dir):
    os.makedirs(net_dir)

for exp in range(cfg.NUM_EXPS):
    ts_lab_path = osp.join(label_dir, 'test_labels_{}.txt'.format(exp))
    for model_name in model_names:
        res_model_dir = osp.join(res_dir, model_name)
        if not osp.exists(res_model_dir):
            os.makedirs(res_model_dir)

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
                accuracy, P, R, F1, conf_mat = net.evaluate(model, test_loader)

                with open(osp.join(res_model_dir,
                                   'res_{}_{}.txt'.format(alg, tr_perc)),
                          'w') as res:
                    np.savetxt(res, conf_mat)
                    res.write('\nACC,P,R,F1\n')
                    res.write('{},{},{},{}\n'.format(accuracy, P, R, F1))
                # print('Final accuracy: {}'.format(net_accuracy))
