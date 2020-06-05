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

    def sample(self, batch_size, num_points):
        k = (batch_size, num_points)
        r = torch.normal(0., 1., size=k)
        x = MultivariateNormal(torch.zeros(3), torch.eye(3)).rsample(k)
        norm = .5 * torch.norm(x, dim=2).log()
        return torch.exp(self.m + self.sigma * r).reshape(batch_size, -1, 1) * x / norm.reshape(batch_size, -1, 1)

    def mle(self, z):
        pi = self.pi
        n = z.size(1)
        m = self.m
        sigma = self.sigma
        norm = self.norm(z)

        return (n * (pi.log() + sigma.log()) +
                3 * norm.sum(1, keepdim=True) +
                ((norm - m) ** 2).sum(1, keepdim=True) / (2 * sigma ** 2))


class SphereScheduler:
    def __init__(self, model, init_m, init_sigma, size):
        self.__sphere = model.sphere
        self.__size = size
        self.__sigma_step = (init_sigma - .01) / size
        self.__m_step = (init_m - 0.) / size

    def step(self):
        old_m = self.__sphere.m
        old_sigma = self.__sphere.sigma

        if self.__size > 0:
            self.__sphere.m -= self.__m_step
            self.__sphere.sigma -= self.__sigma_step
            self.__size -= 1

        new_m = self.__sphere.m
        new_sigma = self.__sphere.sigma
        return (old_m, old_sigma), (new_m, new_sigma)
