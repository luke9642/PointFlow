import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np


class Sphere3DSimple:
    def __init__(self, sigma, m):
        self.sigma = sigma
        self.m = m

    def norm(self, X):
        return .5 * torch.einsum('ijk,ijk->ij', X, X).log()

    # def norm(self, X):
    #     return .5 * torch.norm(X, dim=1).log()

    def pdf(self, X):
        return 1. / (2. * (2. * np.pi) ** (3. / 2) * self.sigma * self.norm(X) ** 3.) * torch.exp(
            -(torch.log(self.norm(X)) - self.m) ** 2. / (2. * self.sigma ** 2))

    def sample(self, k):
        s1 = torch.normal(0., 1., size=k)
        s2 = MultivariateNormal(torch.zeros(3), torch.eye(3)).rsample(k)
        return torch.exp(self.sigma * s1).reshape(-1, 1) * s2 / self.norm(s2).reshape(-1, 1)

    def MLE_1(self, z):
        pi = torch.tensor(2 * ((2 * np.pi) ** (3 / 2)), device='cuda')
        n = z.size(1)
        m = self.m
        sigma = self.sigma
        norm = self.norm(z)

        return (n * (pi.log() + sigma.log()) +
                3 * norm.sum(1, keepdim=True) +
                ((norm - m) ** 2).sum(1, keepdim=True) / (2 * sigma ** 2))
