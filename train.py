import argparse
import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torchvision
import torchvision.transforms as transforms

from model import MicroCls
from logger import create_logger

if not os.path.exists('save_model'):
    os.makedirs('save_model')
if not os.path.exists('log'):
    os.makedirs('log')
logger = create_logger('log')


def test_model(model, device, data_loader, loss_func):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx == 0:
                since = time.time()
            elif batch_idx == len(data_loader)-1:
                logger.info('Eval:[{:5.0f}/{:5.0f} ({:.0f}%)] Loss:{:.4f} Acc:{:.4f} Cost time:{:5.0f}s'.format(
                    total,
                    len(data_loader.dataset),
                    100. * batch_idx / (len(data_loader)-1),
                    test_loss/(batch_idx+1),
                    correct / total,
                    time.time()-since))
    model.train()
    acc = correct / \
        total if total != 0 else 0.
    return acc


def train_model(cfg):
    device = torch.device("cuda:{}".format(cfg.gpu_index)
                          if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_dataloader(cfg.batch_size)
    loss_func = build_loss().to(device)
    model = build_model(
        cfg.nh, cfg.depth, 10).to(device)
    if cfg.model_path != '':
        load_model(cfg.model_path, model)
    optimizer = build_optimizer(model, cfg.lr)
    scheduler = build_scheduler(optimizer)
    best_acc = 0.
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx == 0:
                since = time.time()
            elif batch_idx % cfg.display_interval == 0 or (batch_idx == len(train_loader)-1):
                logger.info('Train:[epoch {}/{} {:5.0f}/{:5.0f} ({:.0f}%)] Loss:{:.4f} Acc:{:.4f} Cost time:{:5.0f}s Estimated time:{:5.0f}s'.format(
                    epoch+1,
                    cfg.epochs,
                    total,
                    len(train_loader.dataset),
                    100. * batch_idx / (len(train_loader)-1),
                    train_loss / (batch_idx+1),
                    correct/total,
                    time.time()-since,
                    (time.time()-since)*(len(train_loader)-1) / batch_idx - (time.time()-since)))
            if batch_idx != 0 and batch_idx % cfg.val_interval == 0:
                acc = test_model(
                    model, device, test_loader, loss_func)
                if acc > best_acc:
                    best_acc = acc
                    save_model(cfg.model_type, model, cfg.nh, cfg.depth, 'best',
                               acc)
        if (epoch+1) % cfg.save_epoch == 0:
            acc = test_model(
                model, device, test_loader, loss_func)
            save_model(cfg.model_type, model, cfg.nh, cfg.depth, epoch+1,
                       acc)
        scheduler.step()


def build_model(nh, depth, nclass):
    return MicroCls(nh=nh, depth=depth, nclass=nclass)


def build_loss():
    return CrossEntropyLoss()


def build_optimizer(model, lr=0.001):
    return optim.Adam(model.parameters(), lr,
                      betas=(0.5, 0.999), weight_decay=0.001)


def build_scheduler(optimizer, step_size=200, gamma=0.8):
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma)
    return scheduler


def build_dataloader(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader


def save_model(model_type, model, nh, depth, epoch, acc):
    if epoch == 'best':
        save_path = './save_model/{}_nh{}_depth{}_best_rec.pth'.format(
            model_type, nh, depth)
        if os.path.exists(save_path):
            data = torch.load(save_path)
            if 'model' in data and data['Acc'] > acc:
                return
        torch.save({
            'model': model.state_dict(),
            'nh': nh,
            'depth': depth,
            'Acc': acc},
            save_path)
    else:
        save_path = './save_model/{}_nh{}_depth{}_epoch{}_Acc{:05f}.pth'.format(
            model_type, nh, depth, epoch, acc)
        torch.save({
            'model': model.state_dict(),
            'nh': nh,
            'depth': depth,
            'Acc': acc},
            save_path)
    logger.info('save model to:'+save_path)


def load_model(model_path, model):
    data = torch.load(model_path)
    if 'model' in data:
        model.load_state_dict(data['model'])
        logger.info('Model loaded nh {}, depth {}, Acc {}'.format(
            data['nh'], data['depth'], data['Acc']))


def main():
    parser = argparse.ArgumentParser(description='MicroCls')
    parser.add_argument('--model_path', default='',
                        help='model path')
    parser.add_argument('--model_type', default='micro',
                        help='model type', type=str)
    parser.add_argument(
        '--nh', default=1024, help='feature width, the more complex the picture background, the greater this value', type=int)
    parser.add_argument(
        '--depth', default=2, help='depth, the greater the number of samples, the greater this value', type=int)
    parser.add_argument('--lr', default=0.001,
                        help='initial learning rate', type=float)
    parser.add_argument('--batch_size', default=200, type=int,
                        help='batch size')
    parser.add_argument('--workers', default=0,
                        help='number of data loading workers', type=int)
    parser.add_argument('--epochs', default=500,
                        help='number of total epochs', type=int)
    parser.add_argument('--display_interval', default=50,
                        help='display interval', type=int)
    parser.add_argument('--val_interval', default=200,
                        help='val interval', type=int)
    parser.add_argument('--save_epoch', default=1,
                        help='how many epochs to save the weight', type=int)
    parser.add_argument('--show_str_size', default=10,
                        help='show str size', type=int)
    parser.add_argument('--gpu_index', default=0, type=int,
                        help='gpu index')
    cfg = parser.parse_args()
    train_model(cfg)


if __name__ == '__main__':
    main()
