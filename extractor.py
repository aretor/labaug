import os
import os.path as osp
import pickle


from extract_tools import get_finetune_model, prepare_loader
from feature_extractor import extract_features_train


class Extractor(object):
    def __init__(self, dset, splitting_dir, feat_dir, net_names):
        self.dset = dset
        self.net_names = net_names
        self.splitting_dir = splitting_dir
        self.feat_dir = feat_dir

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
                    extract_features_train(ft_model, set_loader, net_name)

                # store the name of the net, the dataset on which we are going
                # to use it, and the testing accuracy
                net_info = [net_name, labels, fc7_features, fnames]

                pickle_dir = osp.join(self.feat_dir, set_)
                if not osp.exists(pickle_dir):
                    os.makedirs(pickle_dir)

                with open(osp.join(pickle_dir, net_name + '.pickle'), 'wb')\
                        as f:
                    pickle.dump(net_info, f, pickle.HIGHEST_PROTOCOL)
