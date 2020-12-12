from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from models.wideresnet import *
from models.resnet import *


from losses import alp_loss, pgd_loss, trades_loss, normalize

parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--attack', default='pgd')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=77, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', type = float, default=1.0)
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--snap-epoch', type=int, default=5, metavar='N',
                    help='how many batches to test')                    
parser.add_argument('--model-dir', default='./wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--start-freq', default=1, type=int, metavar='N',
                    help='start point')
parser.add_argument('--loss', default='pgd', type=str, 
                    choices=['pgd', 'pgd_he', 'alp', 'alp_he', 'trades', 'trades_he'])
parser.add_argument('--distance', default='l_inf', type=str, help='distance')
parser.add_argument('--m', default=0.2, type=float, help='angular margin')
parser.add_argument('--s', default=15.0, type=float, help='s value')

args = parser.parse_args()

model_dir = "checkpoint/" + args.loss

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **kwargs
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

# train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

train_batches = Batches(train_set, args.batch_size, shuffle=True)
test_batches = Batches(test_set, args.batch_size, shuffle=False)


# load adversarial examples
adv_dir = "adv_examples/{}/".format(args.attack)
train_path = adv_dir + "train.pth" 
test_path = adv_dir + "test.pth"

adv_train_data = torch.load(train_path)
train_adv_images = adv_train_data["adv"]
train_adv_labels = adv_train_data["label"]

adv_test_data = torch.load(test_path)
test_adv_images = adv_test_data["adv"]
test_adv_labels = adv_test_data["label"]

train_adv_set = list(zip(train_adv_images,
    train_adv_labels))

train_adv_batches = Batches(train_adv_set, args.batch_size, shuffle=True)

test_adv_set = list(zip(test_adv_images,
    test_adv_labels))

test_adv_batches = Batches(test_adv_set, args.batch_size, shuffle=True)



LOSS= {
        'pgd': pgd_loss,
        'pgd_he': pgd_loss,
        'alp': alp_loss,
        'alp_he': alp_loss,
        'trades': trades_loss,
        'trades_he': trades_loss,
}

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (batch, adv_batch) in enumerate(zip(train_batches, train_adv_batches)):
        data = batch["input"]
        target = batch["target"]
        
        x_adv = adv_batch["input"]
        y_adv = adv_batch["target"]
        
        optimizer.zero_grad()

        # calculate robust loss
        loss = LOSS[args.loss](model=model,
                           x_natural=data,
                           x_adv=x_adv
                           y=target,
                           y_adv = y_adv,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta,
                           loss=args.loss,
                           distance=args.distance,
                           m = args.m,
                           s = args.s)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def evaluate(model, device, data_loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(normalize(data))
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    return loss, accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model, ResNet18() can be also used here for training
#     if args.loss == 'alp' or args.loss == 'trades' or args.loss == 'pgd':
#         print("normalize False")
#         model = nn.DataParallel(WideResNet()).to(device)
#     else:
#         print("normalize True")
#         model = nn.DataParallel(WideResNet(use_FNandWN = True)).to(device)

    model = resnet18(pretrained=True)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('==============')
        _, train_accuracy = evaluate(model, device, train_batches)
        _, test_accuracy = evaluate(model, device, test_batches)
        _, train_robust_accuracy = evaluate(model, device, train_adv_batches)
        _, test_robust_accuracy = evaluate(model, device, test_adv_batches)

        print("Train Accuracy: ", train_accuracy)
        print("Test Accuracy: ", test_accuracy)
        print("Train Robust Accuracy: ", train_robust_accuracy)
        print("Test Robust Accuracy: ", test_robust_accuracy)
        print('==============')

        # save checkpoint
        if (epoch >= args.start_freq) and (epoch % args.save_freq == 0):
            torch.save(model.module.state_dict(),
                       os.path.join(model_dir, 'res18-epoch{}.pt'.format(epoch)))

if __name__ == '__main__':
    main()
