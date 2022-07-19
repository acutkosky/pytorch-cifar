'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, scale='none', id=0, activations=[0], **kwargs):
        super(PreActBlock, self).__init__()
        print("preactblock: scale: ",scale)
        print("preactblock: kwargs: ",kwargs)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )
        self.scale = scale
        self.id = id
        self.activations = activations
        if scale == 'separate' or scale == 'both':
            self.scale_factor = nn.Parameter(torch.zeros(1))
        if scale == 'native':
            with torch.no_grad():
                self.conv2.weight *= 0.00001
        if scale == 'both':
            print("scale is both: ",self.scale)
            with torch.no_grad():
                self.conv2.weight *= 0.001      
        self.printed = 0  


    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        if self.id in self.activations:
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            if self.scale == 'separate' or self.scale == 'both':
                out *= torch.tanh(self.scale_factor)
                wandb.log({
                    f'scales/{self.id}': self.scale_factor.item()
                }, commit=False)
                wandb.log({
                    f'scales/nothing': 0.0
                }, commit=False)
                if self.printed < 10:
                    print("activated!")
                    print("id: ",self.id, " scale: ", self.scale_factor)
                    self.printed += 1
            out += shortcut
        else:
            out = shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, stagewise='all', **kwargs):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.activated_layers = [0]
        self.num_blocks = 0
        self.stagewise = stagewise

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, **kwargs)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        if stagewise == 'all':
            for i in range(1, self.num_blocks+1):
                self.activated_layers.append(i)
            self.unactivated_ids = []
        else:
            self.unactivated_ids = list(range(1, self.num_blocks+1))

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        print("kwargs: ",kwargs)
        for stride in strides:
            self.num_blocks += 1
            layers.append(block(self.in_planes, planes, stride, id=self.num_blocks, activations=self.activated_layers, **kwargs))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def activate(self, to_activate=None):
        if len(self.unactivated_ids) == 0:
            return len(self.activated_layers) - 1
        if to_activate is None:
            to_activate = self.unactivated_ids.pop(-1)#self.activated_layers[-1] + 1
        elif to_activate == -1:
            to_activate = self.unactivated_ids.pop(0)
        elif to_activate <= self.num_blocks and to_activate in self.unactivated_ids:
            self.unactivated_ids.remove(to_activate)
        
        assert(to_activate not in self.activated_layers)
        
        self.activated_layers.append(to_activate)
        
        return len(self.activated_layers) - 1 


    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(**kwargs):
    return PreActResNet(PreActBlock, [2,2,2,2], **kwargs)

def PreActResNetMany(layer_count, **kwargs):
    return PreActResNet(PreActBlock, [layer_count] * 4, **kwargs)

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])


def test():
    net = PreActResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()
