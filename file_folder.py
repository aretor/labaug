import os
import os.path as osp

import torchvision.datasets
from torchvision.datasets import folder


def generate_file(img_dset_path):
    with open(os.path.join('..', img_dset_path), 'r') as dset_file:
        class_dirs = os.listdir(img_dset_path)
        for class_dir in [osp.join(img_dset_path, path) for path in class_dirs]:
            files = os.listdir(class_dir)
            for file in [osp.join(class_dir, path) for path in files]:
                dset_file.write(file)


class PathImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        img, target = super(PathImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, target, path


class FileImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, source_file, sep=' ', root='', transform=None,
                 target_transform=None, loader=folder.default_loader,
                 hard_labels=True):
        with open(source_file, 'r') as f:
            fnames = list(f)

        paths = [osp.join(root, fname.split(',')[0].rstrip()) for fname in fnames]
        if len(fnames[0].split(',')) == 2:
            self.classes = [fname.split(sep)[1] for fname in fnames]
        else:
            self.classes = [fname.split('/')[0] for fname in fnames]

        if hard_labels:
            self.class_to_idx = {class_: idx for idx, class_ in enumerate(sorted(set(self.classes)))}
            self.idx_to_class = {class_: self.class_to_idx[class_] for class_ in self.class_to_idx.keys()}
            self.imgs = list(zip(paths, [self.class_to_idx[class_] for class_ in self.classes]))
        else:
            self.imgs = list(zip(paths, self.classes))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.hard_labels = hard_labels

    def __getitem__(self, index):
        img, target = super(FileImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, target, path
