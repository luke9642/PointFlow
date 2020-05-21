import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np


class Sphere3DSimple:
    pi = torch.tensor(2 * ((2 * np.pi) ** (3 / 2)), device='cuda')

    def __init__(self, sigma, m):
        self.sigma = sigma
        self.m = m

    def norm(self, z):
        return .5 * torch.einsum('ijk,ijk->ij', z, z).log()

    def pdf(self, z):
        return 1. / (2. * (2. * np.pi) ** (3. / 2) * self.sigma * self.norm(z) ** 3.) * torch.exp(
            -(torch.log(self.norm(z)) - self.m) ** 2. / (2. * self.sigma ** 2))

    def sample(self, k):
        s1 = torch.normal(0., 1., size=k)
        s2 = MultivariateNormal(torch.zeros(3), torch.eye(3)).rsample(k)
        return torch.exp(self.m + self.sigma * s1).reshape(-1, 1) * s2 / self.norm(s2).reshape(-1, 1)

    def mle(self, z):
        pi = self.pi
        n = z.size(1)
        m = self.m
        sigma = self.sigma
        norm = self.norm(z)

        return (n * (pi.log() + sigma.log()) +
                3 * norm.sum(1, keepdim=True) +
                ((norm - m) ** 2).sum(1, keepdim=True) / (2 * sigma ** 2))
