import sys
import os.path as osp
from tqdm import tqdm
import torch
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def train(net, net_type, train_loader, val_loader, optimizer, criterion, epochs, out_dir, ind=None):
    net.train()
    if ind is not None:
        net_name = osp.join(out_dir, '{}_{}.pth'.format(net_type, ind))
    else:
        net_name = osp.join(out_dir, net_type + '.pth')

    print("Validating results in: start")
    sys.stdout.flush()
    accuracy = evaluate(net, val_loader)[0]
    print(accuracy)

    for epoch in range(epochs):
        net.train()
        print('{} --------- Epoch: {}'.format(net_type, epoch))
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, labels, _ = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            if net_type == 'inception':
                outputs = outputs[0] + outputs[1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
        print('[%d] loss: %.16f' % (epoch + 1, running_loss / len(train_loader)))
        print("Validating results in: {}-th epoch".format(epoch))
        sys.stdout.flush()
        accuracy = evaluate(net, val_loader)[0]
        print(accuracy)

        torch.save(net.state_dict(), net_name)

    return net_name


def evaluate(net, test_loader):
    net.eval()
    correct = total = 0

    predictions, gts = [], []
    for data in tqdm(test_loader, total=len(test_loader)):
        inputs, labels, _ = data
        inputs = Variable(inputs).cuda()
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, dim=1)
        predictions.append(predicted)
        gts.append(labels)

        # total += labels.size(0)
        # correct += (predicted == labels).sum()

    predictions = torch.cat(predictions, dim=0).cpu().numpy()
    gts = torch.cat(gts, dim=0).numpy()

    accuracy = (predictions == gts).sum() / float(len(gts))
    P, R, F1, _ = precision_recall_fscore_support(gts, predictions)
    conf_mat = confusion_matrix(gts, predictions)

    return accuracy, P, R, F1, conf_mat
