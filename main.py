'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys

from models import *
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf

import freeopt
import perpopt


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--wandb_project', default=None, help='set wandb project default loads from wand login setting in environment variable')
parser.add_argument('--scale', type=str, default='none', choices=['none', 'native', 'separate', 'both'])
parser.add_argument('--threshold', type=float, default=0.01)
parser.add_argument('--stagewise', type=str, default='all', choices=['all', 'forward', 'backward'])
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--layer_count', type=int, default=2)
parser.add_argument('--retrain', type=str, default='no', choices=['no', 'first', 'last'])
parser.add_argument('--residual', type=str, default='yes', choices=['no', 'yes'])
parser.add_argument('--retrain_addition', type=int, default=2)
parser.add_argument('--arch', default='resnet18', choices=['resnet18', 'preactresnet18', 'preactresnetmany'])
parser.add_argument('--wd', default=5e-4, type=float)
parser.add_argument('--tag', default='none')

parser.add_argument('--optmain', default='sgd', choices=['sgd', 'free', 'sgd_nom', 'scalefree', 'perpopt', 'adamw'])

parser.add_argument('--optfollow', default='sgd', choices=['sgd', 'free', 'sgd_nom', 'scalefree', 'perpopt', 'adamw'])
parser.add_argument('--optreset', default='no', choices=['no', 'yes'])
parser.add_argument('--ministeps', default=1, type=int)
parser.add_argument('--memorize', default='no', choices=['no', 'yes'])
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--config', default=None)
parser.add_argument('--zerofinal', default='no', choices=['no','yes'])

args = parser.parse_args()
args = OmegaConf.create(vars(args))

if args.config is not None:
    conf = OmegaConf.load(args.config)
    args = OmegaConf.merge(args, conf)

torch.manual_seed(args.manual_seed)

args_dict = OmegaConf.to_container(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.arch == 'resnet18':
    net = ResNet18(args.scale)
if args.arch == 'preactresnet18':
    net = PreActResNet18(**args_dict)
if args.arch == 'preactresnetmany':
    net = PreActResNetMany(**OmegaConf.to_container(args))#layer_count=args.layer_count, scale=args.scale, stagewise=args.stagewise, residual=args.residual)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=0.9, weight_decay=args.wd)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch, examples, it_total, optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        ministeps = args.ministeps
        memorize = args.memorize
        if memorize == 'yes':
            ministeps = max(ministeps, 20)
        for s in range(ministeps+1):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            train_loss_now = loss.item()
            _, predicted = outputs.max(1)
            correct_now = predicted.eq(targets).sum().item()

            if s == 0:
                train_loss += train_loss_now
                total += targets.size(0)
                correct += correct_now
                examples += targets.size(0)
                it_total += 1
                wandb.log({
                    'examples': examples,
                    'epoch': epoch,
                    'train/accuracy': correct/total,
                    'train/loss': train_loss/(batch_idx+1)
                    },
                    step = it_total)
            wandb.log({
                'step_per_batch': s,
                'after_ministep_accuracy': correct_now/targets.size(0)
            },
                step = it_total
            )
            if s >= ministeps:
                break
            if correct_now == targets.size(0) and args.memorize == 'yes':
                break

            optimizer.step()


        pbar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return examples, it_total


def test(epoch, it_total):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_iter = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(testloader), total=len(testloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            test_iter += 1

            pbar.set_description('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = correct/total

    wandb.log({
        'test/accuracy': correct/total,
        'test/loss': test_loss/test_iter,
        },
        step = it_total)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return correct/total

examples = 0
it_total = 0
activate_best_acc = 0
patience = args.patience
bad_epochs = 0
activations = 0
epoch_count = args.epochs
retrain_epochs = 100

def run_training(start_epoch, epoch_count, examples, it_total, activate_best_acc, patience, activations, opttype):
    print(f"run_training args: {(start_epoch, epoch_count, examples, it_total, activate_best_acc, patience, activations, opttype)}")

    if args.onlylinear:
        parameters = net.module.linear.parameters()
    else:
        parameters = net.parameters()
    if opttype == 'sgd':
        optimizer = optim.SGD(parameters, lr=args.lr,
                            momentum=0.9, weight_decay=args.wd)
    elif opttype == 'free':
        optimizer = freeopt.FreeOpt(parameters, lr=args.lr)
    elif opttype == 'sgd_nom':
        optimizer = optim.SGD(parameters, lr=args.lr,
                            momentum=0.0, weight_decay=args.wd)
    elif opttype == 'scalefree':
        optimizer =freeopt.ScaleFree(parameters, epsilon=args.lr, wd=args.wd)
    elif opttype == 'perpopt':
        optimizer = perpopt.PerpOpt(parameters, lr=args.lr, wd=args.wd, **OmegaConf.to_container(args)['perpopt_conf'])
    elif opttype == 'adamw':
        optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_count)

    print(f"running for: {list(range(start_epoch, start_epoch+epoch_count))} epochs...")
    bad_epochs = 0
    for epoch in range(start_epoch, start_epoch+epoch_count):
        examples, it_total = train(epoch, examples, it_total, optimizer)
        acc = test(epoch, it_total)
        if acc < (activate_best_acc + args.threshold):
            bad_epochs += 1
            # print(f'a bad epoch: acc: {acc}, best_acc: {best_acc}, bad_epochs: {bad_epochs}, epoch count: {epoch}')
        else:
            activate_best_acc = acc
            bad_epochs = 0
            # print(f'a good epoch: acc: {acc}, best_acc: {best_acc}, bad_epochs: {bad_epochs}, epoch count: {epoch}')
        wandb.log({
            'best_acc_threshold': activate_best_acc + args.threshold,
            },
            step = it_total)
        wandb.log({
            'acc': acc,
            },
            step = it_total)
        wandb.log({
            'bad_epochs': bad_epochs,
            },
            step = it_total)
        if bad_epochs > patience and args.stagewise != 'all':
            if args.retrain == 'no':
                if args.stagewise == 'forward':
                    activations = net.module.activate()
                if args.stagewise == 'backward':
                    activations = net.module.activate(-1)
            # activations += 1
            bad_epochs = 0
        activations = len(net.module.activated_layers) - 1 
        wandb.log({
            'activations': activations,
            },
            step = it_total)
        scheduler.step()
        if args.optreset == 'yes':
            optimizer.reset()
    return examples, it_total, activate_best_acc, activations


if args.retrain != 'no':
    while len(net.module.unactivated_ids)>(args.layer_count * args.retrain_addition):
        print(f"calling activate because unactivated ids is : {net.module.unactivated_ids}")
        if args.retrain == 'first':
            net.module.activate()
        else:
            net.module.activate(-1)
print(f"activated: {net.module.activated_layers}, unactivead: {net.module.unactivated_ids}")


wandb.init(project=args.wandb_project)
wandb.config.update(args_dict)


examples, it_total, activate_best_acc, activations = run_training(start_epoch, epoch_count,
                                        examples, it_total, activate_best_acc, patience, activations, args.optmain)

print("finished initial training!")

if args.retrain != 'no':
    while (len(net.module.unactivated_ids)>0):
        net.module.activate()


    run_training(epoch_count, retrain_epochs, examples, it_total, activate_best_acc, patience, activations, args.optfollow)