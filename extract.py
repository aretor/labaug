import os
import os.path as osp
import pickle

import torch.random
import torch.cuda.random

import config as cfg
import feature_extractor as fe
from extract_tools import prepare_loader, get_finetune_model


def main():
    torch.random.manual_seed(314)
    torch.cuda.random.manual_seed(314)
    dset_name = 'caltech'
    net_models = ['resnet18', 'densenet121']
    batch_size = 1
    feat_dir = osp.join(cfg.DATA_DIR, dset_name, cfg.FEATURE_DIR)

    dataset = cfg.DSETS[dset_name]
    dset_src, stats, nr_classes = dataset['src'], dataset['stats'], dataset['nr_classes']

    for exp in xrange(cfg.NUM_EXPS):
        for set_ in ('train', 'test'):
            set_pname = osp.join(cfg.DATA_DIR, dset_name, cfg.SPLITTING_DIR, set_ + '_' + str(exp) + '.txt')
            # set_size = size[set_]

            for net_model in net_models:
                print('exp: ' + str(exp), set_, net_model + ' - ' + str(nr_classes) + ' classes')
                ft_model = get_finetune_model(net_model, nr_classes)
                set_loader = prepare_loader(set_pname, dset_src, stats, batch_size, False)

                fc7_features, labels, net, fnames = fe.extract_features_train(ft_model, set_loader,
                                                                              dense='densenet' in net_model)

                # store the name of the net, the dataset on which we are going to use it, and the testing accuracy
                net_info = [net_model, labels, fc7_features, fnames]

                pickle_dir = osp.join(feat_dir, set_)
                if not osp.exists(pickle_dir):
                    os.makedirs(pickle_dir)
                with open(osp.join(pickle_dir, net_model + '_' + str(exp) + '.pickle'), 'wb') as f:
                    pickle.dump(net_info, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
