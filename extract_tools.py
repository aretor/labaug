import torch
import torchvision
import torch.nn as nn
import torch.utils.data

import config as cfg
from file_folder import FileImageFolder


def get_finetune_model(net, nr_classes):
    model = cfg.NETS[net](pretrained=True)
    if 'resnet' in net:
        model.fc = nn.Linear(model.fc.in_features, nr_classes)
    elif 'densenet' in net:
        model.classifier = nn.Linear(model.classifier.in_features, nr_classes)

    return model


# TODO: change collate_fn to work correctly with FileImageFolder (the path is returned as a list instead of str)
def prepare_loader(dataset_path, img_root, stats, batch_size, shuffle, sep='',
                   hard_labels=True):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(stats[0], stats[1], stats[2]),
                                         std=(stats[3], stats[4], stats[5]))
    ])

    if not hard_labels:
        target_transform = lambda labs: torch.Tensor([float(l) for l in labs.rstrip().split(' ')])
    else:
        target_transform = None

    val = FileImageFolder(dataset_path, root=img_root, transform=transform,
                          sep=sep, hard_labels=hard_labels,
                          target_transform=target_transform)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=12,
                                             pin_memory=True)
    return val_loader
