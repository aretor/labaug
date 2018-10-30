import os
import os.path as osp
import random


class Splitter(object):
    def __init__(self, dset, splitting_dir):
        self.dset = dset
        self.splitting_dir = splitting_dir

    def __call__(self, *args, **kwargs):
        """ Given a dataset generate the splitting of training and test set,
            the splitting will be put in an appropriate
            folder named after the dset_name parameter

            Input:
            tr_frac: the fraction of observation in the training set (remaining
            in the test set(
            exts: list of allowed file extensions, in the format .<name of
            extenstion. Return files with the specified extensions, (all files
            if exts=None)
        """
        tr_frac = kwargs.pop('tr_frac', 0.7)
        exts = kwargs.pop('exts', None)

        if not osp.exists(self.splitting_dir):
            os.makedirs(self.splitting_dir)

        with open(osp.join(self.splitting_dir, 'train.txt'), 'w') as train, \
                open(osp.join(self.splitting_dir, 'test.txt'), 'w') as test:
            for class_dir in os.listdir(self.dset['src']):
                fnames = os.listdir(osp.join(self.dset['src'], class_dir))
                fnames = [f for f in fnames if
                          (exts is None) or (osp.splitext(f)[1] in exts)]
                random.shuffle(fnames)

                split = int(tr_frac * len(fnames))
                assert 0 < split < len(fnames), \
                    "unbalanced split in: " + class_dir +\
                    ". try change \'tr_frac\'"
                for line in fnames[:split]:
                    train.write(osp.join(class_dir, line + "\n"))

                for line in fnames[split:]:
                    test.write(osp.join(class_dir, line + "\n"))
