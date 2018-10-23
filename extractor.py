import os
import os.path as osp
import pickle
import torch
from torch import nn
import torch.nn.functional as F

from extract_tools import get_finetune_model, prepare_loader


class Extractor(object):
    def __init__(self, dset, splitting_dir, feat_dir, net_names):
        self.dset = dset
        self.net_names = net_names
        self.splitting_dir = splitting_dir
        self.feat_dir = feat_dir

    def _extract_features(self, net, loader, model_name):
        net.eval()
        layers = list(net.children())
        net = nn.Sequential(*layers[:-1])

        features = torch.zeros(len(loader), layers[-1].in_features)
        labels_ = torch.zeros(len(loader), 1)
        fnames = []

        for k, data in tqdm(enumerate(loader), total=len(loader)):
            inputs, labels, path = data
            inputs, labels = torch.autograd.Variable(
                inputs).cuda(), torch.autograd.Variable(labels).cuda()
            outputs = F.relu(net(inputs))
            if 'densenet' in model_name:
                outputs = torch.squeeze(F.avg_pool2d(outputs, 7))
            features[k, :] = outputs.data
            labels_[k, :] = labels.data
            fnames.append(path[0])

        fc7_features = features.numpy()
        labels = labels_.numpy()
        return fc7_features, labels, net, fnames

    def __call__(self, *args, **kwargs):
        """ Extract the features """
        batch_size = kwargs.pop('batch_size_fe', 1)

        for set_ in ('train', 'test'):
            for net_name in self.net_names:
                print('{}, {}, {} classes'.format(set_, net_name,
                                                  self.dset['nr_classes']))
                ft_model = get_finetune_model(net_name, self.dset['nr_classes'])
                set_loader = prepare_loader(
                    osp.join(self.splitting_dir, set_ + '.txt'),
                    self.dset['src'], self.dset['stats'], batch_size, False)

                fc7_features, labels, model, fnames = \
                    self._extract_features(ft_model, set_loader, net_name)

                net_info = [net_name, labels, fc7_features, fnames]

                pickle_dir = osp.join(self.feat_dir, set_)
                if not osp.exists(pickle_dir):
                    os.makedirs(pickle_dir)

                with open(osp.join(pickle_dir, net_name + '.pickle'), 'wb')\
                        as f:
                    pickle.dump(net_info, f, pickle.HIGHEST_PROTOCOL)
