# -*- coding: utf-8 -*-
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.input_drop = nn.Dropout2d(0.2)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0)
        self.conv3_drop = nn.Dropout2d(0.5)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(self.input_drop(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv3_drop(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        x = F.log_softmax(x)
        return x


# set model to train mode
def train(epoch, model, optimizer, train_loader):
    model.train()
    loss_sum = 0
    num_instances = train_loader.dataset.data_tensor.size()[1]
    for batch_idx, (data, target) in enumerate(train_loader):
        # skip this if you have no GPU
        data = data.cuda()
        target = target.cuda()
        # set autograd variables
        data, target = th.autograd.Variable(data), th.autograd.Variable(target)
        # zero gradients for safety
        optimizer.zero_grad()
        # feedforward
        output = model(data)
        # compute loss
        loss = th.nn.functional.nll_loss(output, target)
        loss_sum += loss.data[0]
        # back-prop loss
        loss.backward()
        # oprimizer run
        optimizer.step()
    print('Train Epoch %i, Loss %f'%(epoch, loss_sum/num_instances))
    
#     
def test(epoch, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = th.autograd.Variable(data, volatile=True), th.autograd.Variable(target)
        output = model(data)
        test_loss += th.nn.functional.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    accuracy = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return accuracy