import random
import torch
import torch.nn as nn


def deactivate_transferstyle(m):
    if type(m) == TransferStyle:
        m.set_activation_status(False)


def activate_transferstyle(m):
    if type(m) == TransferStyle:
        m.set_activation_status(True)



class TransferStyle(nn.Module):
    """MixStyle (w/ domain prior).
    The input should contain two equal-sized mini-batches from two distinct domains.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha

        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps})'

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x):
        """
        For the input x, the first half comes from one domain,
        while the second half comes from the other domain.
        """
        if not self.training or not self._activated:
            return x
        #####=================================================
        ##### mix style
        #####=================================================
        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.arange(B - 1, -1, -1)  # inverse index
        perm_b, perm_a = perm.chunk(2)
        perm_b = perm_b[torch.randperm(B // 2)]
        perm_a = perm_a[torch.randperm(B // 2)]
        perm = torch.cat([perm_b, perm_a], 0)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix

        #####=================================================
        ##### transfer style
        #####=================================================
       #  if random.random() > self.p:
       #      return x
       #
       #  B = x.size(0)
       #
       #  mu = x.mean(dim=[2, 3], keepdim=True)
       #  var = x.var(dim=[2, 3], keepdim=True)
       #  sig = (var + self.eps).sqrt()
       #  mu, sig = mu.detach(), sig.detach()
       #  x_normed = (x - mu) / sig
       #
       #  #lmda = self.beta.sample((B, 1, 1, 1))
       #  lmda=torch.zeros(B, 1, 1, 1)
       #  lmda = lmda.to(x.device)
       #
       #  perm = torch.arange(B - 1, -1, -1)  # inverse index
       #  perm_b, perm_a = perm.chunk(2)
       #  perm_b = perm_b[torch.randperm(B // 2)]
       #  perm_a = perm_a[torch.randperm(B // 2)]
       #  perm = torch.cat([perm_b, perm_a], 0)
       #
       #  mu2, sig2 = mu[perm], sig[perm]
       #  mu_mix = mu * lmda + mu2 * (1 - lmda)
       #  sig_mix = sig * lmda + sig2 * (1 - lmda)
       #
       #  x_target = (x_normed * sig_mix + mu_mix)+x
       #  x_new = torch.cat([x[0:B // 2], x_target[B // 2:]],0)
       # # x_new = torch.cat([x[0:(B*3) // 4], x_target[(B*3) // 4:]], 0)
       #
       #  return x_new