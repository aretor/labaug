import os.path as osp
import torch
from torch.autograd import Variable


def train(net, net_type, train_loader, val_loader, optimizer, criterion, epochs, out_dir, ind):
    net.train()
    net_name = osp.join(out_dir, net_type + '_{}.pth'.format(ind))

    print("Validating results in: start")
    new_accuracy = evaluate(net, val_loader)
    print(new_accuracy)

    for epoch in range(epochs):
        net.train()
        print(net_type + ' --------- ' + 'Epoch: ' + str(epoch))
        running_loss = 0.0
        for i, data in enumerate(train_loader):
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
            if i == 10:  # print stats after 10 mini batches for each epoch
                print('[%d, %5d] loss: %.16f' % (epoch + 1, i + 1, running_loss / 10.0))
            running_loss = 0.0

        print("Validating results in: {}-th epoch".format(epoch))
        new_accuracy = evaluate(net, val_loader)
        print(new_accuracy)

        torch.save(net.state_dict(), net_name)

    return net_name


def evaluate(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    for data in test_loader:
        inputs, labels, _ = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(torch.autograd.Variable(inputs))
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    accuracy = correct / float(total)
    return accuracy
