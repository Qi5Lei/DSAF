# Written by Yixiao Ge

import torch
import torch.nn as nn


def Wasserstein(mu, sigma, idx1, idx2): #input shape (b,c), output: shape[b]
    p1 = torch.sum(torch.pow((mu[idx1] - mu[idx2]),2),1)
    p2 = torch.sum(torch.pow(torch.pow(sigma[idx1],1/2) - torch.pow(sigma[idx2], 1/2),2) , 1)
    return p1+p2

class DSBN(nn.Module):
    def __init__(
        self,
        num_features,
        num_domains,
        batchnorm_layer=nn.BatchNorm2d,
        eps=1e-5,
        momentum=0.1,
        target_bn_idx=-1,
        weight_requires_grad=True,
        bias_requires_grad=True,
    ):
        super(DSBN, self).__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.target_bn_idx = target_bn_idx
        self.batchnorm_layer = batchnorm_layer

        dsbn = [batchnorm_layer(num_features, eps=eps, momentum=momentum) # track_running_stats=False
                    for _ in range(num_domains)]
        for idx in range(num_domains):
            dsbn[idx].weight.requires_grad_(weight_requires_grad)
            dsbn[idx].bias.requires_grad_(bias_requires_grad)
        self.dsbn = nn.ModuleList(dsbn)

    def forward(self, x):
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_test(x)

    def _forward_train(self, x):
        bs = x.size(0)
        # print('$$'*100)
        # print(bs)
        # print('$$' * 100)
        assert bs % self.num_domains == 0, "the batch size should be times of BN groups"

        #split = torch.split(x, [int(bs // self.num_domains), int(bs // (self.num_domains*2)), int(bs // (self.num_domains*2))], 0)
        split = torch.split(x, int(bs // self.num_domains), 0)
        out = []
        for idx, subx in enumerate(split):
            out.append(self.dsbn[idx](subx.contiguous()))
        return torch.cat(out, 0)

    def _forward_test(self, x):
        # Default: the last BN is adopted for target domain
        return self.dsbn[self.target_bn_idx](x)


###############
##############
class IBN_MODULE_share_in_affine(nn.Module):
    def __init__(
        self,
        num_features,
        num_domains,
        target_bn_idx=-1,
        eps=1e-5,
        momentum=0.1,
        weight_requires_grad=True,
        bias_requires_grad=True,
    ):
        super(IBN_MODULE_share_in_affine, self).__init__()
        self.num_features = num_features
        self.num_domains = num_domains
        self.target_bn_idx = target_bn_idx
        half1 = int(num_features / 2)
        self.half = half1

        self.IN = nn.InstanceNorm2d(half1, affine=True)#True
        dsbn = [nn.BatchNorm2d(half1, eps=eps, momentum=momentum) for _ in range(num_domains)]
        for idx in range(num_domains):
            dsbn[idx].weight.requires_grad_(weight_requires_grad)
            dsbn[idx].bias.requires_grad_(bias_requires_grad)
        self.dsbn = nn.ModuleList(dsbn)

    def forward(self, x):
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_test(x)

    def _forward_train(self, x):
        bs = x.size(0)
        # print('$$'*100)
        # print(bs)
        # print('$$' * 100)
        assert bs % self.num_domains == 0, "the batch size should be times of BN groups"

        split = torch.split(x, self.half, 1)
        out_in = self.IN(split[0].contiguous())

        split_bn = torch.split(split[1], int(bs // self.num_domains), 0)
        out_bn = []
        for idx, subx in enumerate(split_bn):
            #out_bn.append(self.DSBN.dsbn[idx](subx.contiguous()))
            out_bn.append(self.dsbn[idx](subx.contiguous()))
        out_bn = torch.cat(out_bn, 0)
        return torch.cat((out_in, out_bn), 1)

    def _forward_test(self, x):
        # Default: the last BN is adopted for target domain
        sub_split = torch.split(x, self.half, 1)
        out1 = self.IN(sub_split[0].contiguous())
        out2 = self.dsbn[self.target_bn_idx](sub_split[1].contiguous())
        sub_out = torch.cat((out1, out2), 1)
        return sub_out
