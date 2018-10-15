import os
import os.path as osp
import shutil
import distutils.dir_util
import random

from skimage import io

import config as cfg
root = 'Datasets'


def prepare_sun_dataset(dataset_root):
    distutils.dir_util.remove_tree('misc')
    distutils.dir_util.remove_tree('outliers')

    for letter_dir in map(lambda dir: osp.join(dataset_root, dir), os.listdir(dataset_root)):
        for class_dir in map(lambda dir: osp.join(letter_dir, dir), os.listdir(letter_dir)):
            if osp.isdir(class_dir):
                for class_obj in map(lambda dir: osp.join(class_dir, dir), os.listdir(class_dir)):
                    if osp.isdir(class_obj):
                        for file in map(lambda file_: osp.join(class_obj, file_), os.listdir(class_obj)):
                            shutil.copy(file, class_dir)
                        distutils.dir_util.remove_tree(class_obj)
            else:
                os.remove(class_dir)
        distutils.dir_util.copy_tree(letter_dir, dataset_root)
        distutils.dir_util.remove_tree(letter_dir)


def gen_tr_ts_files(dset_name, tr_frac=0.7, n=1, exts=None):
    """ Given a dataset generate the splitting of training and test set, the splitting will be put in an appropriate
        folder named after the dset_name parameter
        dataset_root: the dataset root folder
        dataset_name: the name of the dataset
        tr_frac: the fraction of observation in the training set (remaining in the test set(
        n: number of splitting to be generated
        exts: list of allowed file extensions, in the format .<name of extenstion. Return files with the specified
              extensions, (all files if exts=None)
    """

    dset_source = osp.expanduser(cfg.DSETS[dset_name]['src'])
    dset_target = osp.join(exp_dir, dset_name)

    if not osp.exists(dset_target):
        os.makedirs(dset_target)

    splitting_dir = osp.join(dset_target, cfg.SPLITTING_DIR)
    if not osp.exists(splitting_dir):
        os.makedirs(splitting_dir)

    with open(osp.join(splitting_dir, 'train.txt'), 'w') as train, \
         open(osp.join(splitting_dir, 'test.txt'), 'w') as test:
        for class_dir in os.listdir(dset_source):
            fnames = os.listdir(osp.join(dset_source, class_dir))
            fnames = [f for f in fnames if (exts is None) or (osp.splitext(f)[1] in exts)]
            random.shuffle(fnames)

            split = int(tr_frac * len(fnames))
            for line in fnames[:split]:
                train.write(osp.join(class_dir, line + "\n"))

            for line in fnames[split:]:
                test.write(osp.join(class_dir, line + "\n"))


def split(dataset_name, tr_perc, n=1):
    source = osp.join(root, dataset_name)

    for i in xrange(n):
        for folder in os.listdir(source):
            source_folder = osp.join(source, folder)
            train_folder = osp.join(root, 'caltech', 'train_' + str(i), folder)
            test_folder = osp.join(root, 'caltech', 'test_' + str(i), folder)

            try:
                os.makedirs(train_folder)
                os.makedirs(test_folder)
            except IOError:
                pass

            # if not osp.exists(train_folder): os.makedirs(train_folder)
            # if not osp.exists(test_folder): os.makedirs(test_folder)

            files = os.listdir(source_folder)
            random.shuffle(files)

            split = int(tr_perc * len(files))

            for file in files[:split]:
                shutil.copy(osp.join(source_folder, file), train_folder)

            for file in files[split:]:
                shutil.copy(osp.join(source_folder, file), test_folder)


def gen_gtg_dataset(dset_name, data_fname, lab_perc):
    source = osp.join(root, dset_name)

    with open(data_fname, 'r') as f:
        for line in f:
            fname, lab = line.split(' ')
            lab = lab[:-1]
            dst_pname = osp.join(root, dset_name, 'train_labelled' + str(lab_perc), lab)
            try:
                os.makedirs(dst_pname)
            except OSError:
                pass
            shutil.copy(osp.join(source, fname), dst_pname)


def compute_img_dataset_stats(dataset_file, src_folder=None):
    """
    Compute the mean and standard deviation of R,G,B channels of an image dataset
    :param dataset_file: path to a filename containing the image filenames of the dataset
    :param src_folder: path to the folder containing the image folders
    :return: respectively R, G and B means, and R, G and B standard deviations
    """
    if src_folder is None:
        src_folder = ''

    R, G, B = OnlineStats(), OnlineStats(), OnlineStats()
    with open(dataset_file, 'r') as f:
        for fname in map(lambda fname_: osp.join(src_folder, fname_.rstrip()), f):
            img = io.imread(fname).reshape(-1, 3)
            R.push(img[:, 0]), G.push(img[:, 1]), B.push(img[:, 2])

    return R.mean(), G.mean(), B.mean(), R.std(), G.std(), B.std()


if __name__ == '__main__':
    random.seed(2718)
    dset_name = 'caltech'
    gen_tr_ts_files(cfg.DSETS[dset_name]['src'], dset_name, tr_frac=0.7, exts=['.jpg', '.jpeg', '.png'])
    # print(compute_img_dataset_stats('dataset_files/indoors/train_0.txt', 'Datasets/indoors'))
