

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import wandb

SMALL_VALUE=1e-5

class CondNormDiag(nn.Module):

    def __init__(self, shape, bias=True):
        super(CondNormDiag, self).__init__()

        self.bias = bias
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('variance', torch.zeros(shape))
        self.register_buffer('count', torch.zeros(1))

        self.scale = nn.Parameter(torch.ones(shape)) 
        if self.bias:
            self.bias = nn.Parameter(torch.zeros(shape))


    def train(self, mode):
        super(CondNormDiag, self).train(mode)


    def forward(self, X):

        # we're going to do a kind of 'improper' update style thing even if 
        # in eval mode:
        with torch.no_grad():
            new_count = self.count + 1
            new_mean = self.mean + (X - self.mean)/new_count
            new_variance = self.variance + ((X - self.mean)**2 - self.variance)/new_count

        X = self.scale * (X - new_mean)/torch.sqrt(new_variance + SMALL_VALUE)
        if self.bias:
            X += self.bias



        # update statistics if in training mode
        if self.training:
            with torch.no_grad():
                self.count.copy_(new_count)
                self.mean.copy_(new_mean)
                self.variance.copy_(new_variance)

        

        return X
        



class CondNorm2d(nn.Module):

    def __init__(self, channels, use_bias=True, affine=True, eps=SMALL_VALUE):
        super(CondNorm2d, self).__init__()

        self.channels = channels
        self.eps = eps
        self.use_bias = use_bias
        self.momentum = 0.0
        self.register_buffer('mean', torch.zeros((1, channels, 1, 1)))
        self.register_buffer('variance', torch.zeros((1, channels, 1, 1)))
        self.register_buffer('count', torch.zeros(1))
        self.affine = affine
        if affine:
            self.scale = nn.Parameter(torch.ones((1, channels, 1, 1)))
            if self.use_bias:
                self.bias = nn.Parameter(torch.zeros((1, channels, 1, 1)))


    def train(self, mode):
        super(CondNorm2d, self).train(mode)


    def forward(self, X):

        # we're going to do a kind of 'improper' update style thing even if 
        # in eval mode:
        B, C, H, W = X.shape
        per_example_count_increment = B* H * W #+ torch.sqrt(self.count)
        # with torch.no_grad():
        per_example_new_count = self.count + per_example_count_increment

        momentum = torch.clamp(per_example_count_increment/per_example_new_count, min=self.momentum)
        # X_mean = torch.einsum('b c h w -> c', X)
        X_mean = torch.mean(X, dim=[0,2,3], keepdim=True)
        # X_mean = torch.einsum('b c h w -> c', X) / count_increment
        

        per_example_mean = self.mean * (1- momentum) + momentum * X_mean 


        X_variance = torch.mean(X**2, dim=[0,2,3], keepdim=True)
        # new_mean = self.mean + (X_mean - self.mean) * count_increment/new_count

        # expanded_mean = einops.rearrange(new_mean, '(b c h w) -> b c h w', b=1, h=1, w=1)
        # X_variance = torch.einsum('b c h w -> c', (X - expanded_mean)**2)


        # expanded_X_mean = einops.rearrange(X_mean, '(b c h w) -> b c h w', b=1, h=1, w=1)
        # X_variance = torch.einsum('b c h w -> c', (X - expanded_X_mean)**2) / count_increment
        # X_variance = torch.mean((X-expanded_X_mean)**2, dim=[0,2,3])

        per_example_variance = self.variance * (1- momentum) + momentum * X_variance

        # new_variance = self.variance + (X_variance - self.variance)*count_increment/new_count

        # expanded_variance = einops.rearrange(new_variance, '(b c h w) -> b c h w', b=1, h=1, w=1)

        # expanded_X_variance = einops.rearrange(X_variance, '(b c h w) -> b c h w', b=1, h=1, w=1)

        X = (X - per_example_mean)*torch.rsqrt(per_example_variance - per_example_mean**2 + self.eps)
        if self.affine:
            X = self.scale * X
            if self.use_bias:
                X += self.bias



        # update statistics if in training mode
        if self.training:
            with torch.no_grad():
                count_increment = B * H * W
                new_count = self.count + count_increment
                momentum = torch.clamp(count_increment/new_count, min=self.momentum)
                full_X_mean = torch.mean(X, dim=[0, 2,3], keepdim=True)
                full_X_variance = torch.mean(X**2, dim=[0, 2,3], keepdim=True)
                new_mean = self.mean * (1- momentum) + momentum * full_X_mean 
                new_variance = self.variance * (1- momentum) + momentum * full_X_variance
                self.count.copy_(new_count)
                self.mean.copy_(new_mean)
                self.variance.copy_(new_variance)

        

        return X
        



if __name__ == '__main__':
    picture = torch.tensor([ [[[0.0,0.0], [0.0,0.0]],
                              [[1.0,1.0], [1.0,1.0]]],
                             [[[1.0,1.0], [1.0,1.0]],
                              [[1.0,1.0], [1.0,1.0]]] ])
    picture2 = torch.tensor([ [[[5.0,0.0], [0.0,0.0]],
                               [[1.0,1.0], [1.0,1.0]]],
                              [[[1.0,1.0], [1.0,1.0]],
                               [[1.0,1.0], [1.0,1.0]]] ])

    B = 10
    C = 10
    H = 10
    N = 10
    picture_big= torch.randn((B, C, H, N))
    picture2_big = torch.randn((B, C, H, N))
    print(picture.shape)

    bn = torch.nn.BatchNorm2d(C)
    cn = CondNorm2d(C)

    bn.train(True)
    cn.train(True)
    # print(bn(picture) - cn(picture))
    # print(bn(picture2) - cn(picture2))

    # print(bn.weight)
    # print(cn.scale)

    sbn = torch.optim.SGD(bn.parameters(), lr=0.1)
    scn = torch.optim.SGD(cn.parameters(), lr=0.1)

    pbn = torch.sum(bn(picture_big))
    pbn = torch.sum(bn(picture_big))
    pbn = torch.sum(bn(picture_big))
    pbn = torch.sum(bn(picture_big))
    pbn = torch.sum(bn(picture_big))
    pbn.backward()
    sbn.step()

    pcn = torch.sum(cn(picture_big))
    pcn = torch.sum(cn(picture_big))
    pcn = torch.sum(cn(picture_big))
    pcn = torch.sum(cn(picture_big))
    pcn = torch.sum(cn(picture_big))
    pcn.backward()
    scn.step()

    # print(bn(picture) - cn(picture))
    print(bn(picture2_big) - cn(picture2_big))



    # print(cn(picture))
    # print(cn(picture2))