import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

class Sphere3DSimple:
    def __init__(self, sigma, m):
        self.sigma = sigma
        self.m = m

    def norm(self, X):
        return torch.norm(X, dim=1).to('cuda')

    def pdf(self, X):
        return 1. / (2. * (2. * np.pi) ** (3. / 2) * self.sigma * self.norm(X) ** 3.) * torch.exp(
            -(torch.log(self.norm(X)) - self.m) ** 2. / (2. * self.sigma ** 2))

    def sample(self, k):
        s1 = torch.normal(0., 1., size=k)
        s2 = MultivariateNormal(torch.zeros(3), torch.eye(3)).rsample(k)
        return torch.exp(self.sigma * s1).reshape(-1, 1) * s2 / self.norm(s2).reshape(-1, 1)

    def MLE_1(self, X):
        pi = torch.tensor([2 * (2 * np.pi) ** (3 / 2)]).to('cuda')
        sigma = torch.tensor([self.sigma]).to('cuda')
        return torch.log(pi) + torch.log(sigma) + 3 * torch.log(self.norm(X)) + 1 / (2 * self.sigma ** 2) * (torch.log(self.norm(X)) - self.m) ** 2
