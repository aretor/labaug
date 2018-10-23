from tqdm import tqdm
import torch
import torch.utils.data
import torch.nn as nn

import torch.nn.functional as F
import extract_tools


def get_net_info(net_processed_name, number_of_classes, nets_and_features):
    net, feature_size = extract_tools.create_net(number_of_classes, nets_and_features, net_processed_name)
    return net, feature_size


def one_hot(labels, number_of_classes):
    len_ = 1 if isinstance(labels, int) else len(labels)
    label_one_hot = torch.zeros(len_, number_of_classes)
    label_one_hot[list(xrange(len_)), labels] = 1.
    return label_one_hot


def extract_features_train(net, train_loader, model_name):
    net.eval()
    layers = list(net.children())
    net = nn.Sequential(*layers[:-1])

    features = torch.zeros(len(train_loader), layers[-1].in_features)
    labels_ = torch.zeros(len(train_loader), 1)
    fnames = []

    for k, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs, labels, path = data
        inputs, labels = torch.autograd.Variable(inputs).cuda(), torch.autograd.Variable(labels).cuda()
        outputs = F.relu(net(inputs))
        if 'densenet' in model_name:
            outputs = torch.squeeze(F.avg_pool2d(outputs, 7))
        features[k, :] = outputs.data
        labels_[k, :] = labels.data
        fnames.append(path[0])

    fc7_features = features.numpy()
    labels = labels_.numpy()
    return fc7_features, labels, net, fnames
