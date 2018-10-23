import sys
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from extract_tools import get_finetune_model, prepare_loader


class Trainer(object):
    def __init__(self, dset, label_dir, net_dir, res_dir, net_names, hard_labels):
        self.dset = dset
        self.net_names = net_names
        self.label_dir = label_dir
        self.net_dir = net_dir
        self.res_dir = res_dir
        self.hard_labels = hard_labels

        self.train_loader = None
        self.test_loader = None

    def train(self, net, net_type, lab_path, epochs, lr, batch_size):
        net.train()
        net_name = osp.join(self.net_dir, net_type + '.pth')
        ts_lab_path = osp.join(self.label_dir, 'test_labels.txt')

        criterion = nn.CrossEntropyLoss() if self.hard_labels else nn.KLDivLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)

        if not self.hard_labels:
            logsoftmax = torch.nn.LogSoftmax(dim=1)

        self.train_loader = prepare_loader(lab_path, '', self.dset['stats'], batch_size, shuffle=True,
                                           sep=',', hard_labels=self.hard_labels)

        self.test_loader = prepare_loader(ts_lab_path, '', self.dset['stats'], batch_size, shuffle=False,
                                          sep=',', hard_labels=self.hard_labels)

        print("Validating results in: start")
        sys.stdout.flush()
        accuracy = self.evaluate(net)[0]
        print(accuracy)

        for epoch in range(epochs):
            net.train()
            print('{} --------- Epoch: {}'.format(net_type, epoch))
            running_loss = 0.0
            for i, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                inputs, labels, _ = data
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                optimizer.zero_grad()
                outputs = net(inputs)
                if net_type == 'inception':
                    outputs = outputs[0] + outputs[1]

                if not self.hard_labels:
                    outputs = logsoftmax(outputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
            sys.stderr.flush()
            print('[%d] loss: %.16f' % (epoch + 1, running_loss / len(self.train_loader)))
            print("Validating results in: %d-th epoch" % epoch)
            sys.stdout.flush()
            accuracy = self.evaluate(net)[0]
            print("Accuracy: %f" % accuracy)

            torch.save(net.state_dict(), net_name)

        return net_name

    def evaluate(self, net):
        net.eval()
        predictions, gts = [], []
        for data in tqdm(self.test_loader, total=len(self.test_loader)):
            inputs, labels, _ = data
            inputs = Variable(inputs).cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            predictions.append(predicted)
            gts.append(labels)
        predictions = torch.cat(predictions, dim=0).cpu().numpy()
        gts = torch.cat(gts, dim=0).numpy()

        accuracy = (predictions == gts).sum() / float(len(gts))
        P, R, F1, _ = precision_recall_fscore_support(gts, predictions)
        conf_mat = confusion_matrix(gts, predictions)

        return accuracy, P, R, F1, conf_mat

    def __call__(self, *args, **kwargs):
        tr_percs = kwargs.pop('tr_percs', [0.02, 0.05, 0.1])
        algs = kwargs.pop('algs', ['gtg', 'svm', 'labels_only'])
        batch_size = kwargs.pop('batch_size_tr', 8)
        epochs = kwargs.pop('epochs', 10)
        lr = kwargs.pop('lr', 1e-4)

        if not osp.exists(self.net_dir):
            os.makedirs(self.net_dir)

        for net_name in self.net_names:
            res_model_dir = osp.join(self.res_dir, net_name)
            if not osp.exists(res_model_dir):
                os.makedirs(res_model_dir)

            for tr_perc in tr_percs:
                for alg in algs:
                    lab_path = osp.join(self.label_dir, alg, net_name, 'labels_{}.txt'.format(tr_perc))

                    model = get_finetune_model(net_name, self.dset['nr_classes'])
                    trained_net = self.train(model, net_name, lab_path, epochs, lr=lr, batch_size=batch_size)

                    model.load_state_dict(torch.load(trained_net))
                    accuracy, P, R, F1, conf_mat = self.evaluate(model)

                    with open(osp.join(res_model_dir, 'res_{}_{}.txt'.format(alg, tr_perc)), 'w') as res:
                        res.write('ACC,P,R,F1\n')
                        res.write('{},{},{},{}\n'.format(accuracy, P, R, F1))

                    np.savetxt(osp.join(res_model_dir, 'conf_mat_{}_{}.txt'.format(alg, tr_perc)), conf_mat)
