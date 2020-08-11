import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np


class Sphere3DSimple:
    def __init__(self, sigma: float, m: float, device: str = 'cuda'):
        self.sigma = sigma
        self.m = m
        self.device = device

    def norm(self, z: torch.Tensor) -> torch.Tensor:
        return .5 * torch.einsum('ijk,ijk->ij', z, z).log()

    def pdf(self, z: torch.Tensor) -> torch.Tensor:
        return 1. / (2. * (2. * np.pi) ** (3. / 2) * self.sigma * self.norm(z) ** 3.) * torch.exp(
            -(torch.log(self.norm(z)) - self.m) ** 2. / (2. * self.sigma ** 2))

    def sample(self, batch_size: int, num_points: int) -> torch.Tensor:
        k = (batch_size, num_points)
        r = torch.normal(0., 1., size=k, device=self.device)
        x = MultivariateNormal(torch.zeros(3, device=self.device), torch.eye(3, device=self.device)).rsample(k)
        norm = torch.norm(x, dim=2)

        return torch.exp(self.m + self.sigma * r).reshape(batch_size, -1, 1) * x / norm.reshape(batch_size, -1, 1)

    def mle(self, z: torch.Tensor) -> torch.Tensor:
        pi = torch.tensor(2 * ((2 * np.pi) ** (3 / 2)), device=self.device)
        n = z.size(1)
        m = self.m
        sigma = self.sigma
        norm = self.norm(z)

        return (n * (pi.log() + np.math.log(sigma)) +
                3 * norm.sum(1, keepdim=True) +
                ((norm - m) ** 2).sum(1, keepdim=True) / (2 * sigma ** 2))

    def __str__(self):
        return f'Sphere: (m: {self.m:.5f}, sigma: {self.sigma:.5f})'


class SphereScheduler:
    def __init__(self, model, size, init_m, init_sigma, target_m=0., target_sigma=.01):
        self.__sphere = model.sphere
        self.__m_step = (init_m - target_m) / size
        self.__sigma_step = (init_sigma - target_sigma) / size
        self.__target_m = target_m
        self.__target_sigma = target_sigma
        self.__finished = False
        print(f'''SphereScheduler
    (init_m: {init_m}, init_sigma: {init_sigma})
    (actual_m: {self.__sphere.m}, actual_sigma: {self.__sphere.sigma})
    (target_m: {self.__target_m}, target_sigma: {self.__target_sigma})
    (m_step: {self.__m_step}, sigma_step: {self.__sigma_step})
''')

    def step(self):
        if self.__finished:
            return

        old_m = self.__sphere.m
        old_sigma = self.__sphere.sigma

        new_m = self.__sphere.m - self.__m_step
        new_sigma = self.__sphere.sigma - self.__sigma_step

        if new_m <= self.__target_m or new_sigma <= self.__target_sigma:
            self.__finished = True
            print('= End of sphere schedule =')
            print(self.__sphere)
            return

        self.__sphere.m = new_m
        self.__sphere.sigma = new_sigma

        print(f'm {old_m:.5f} -> {new_m:.5f} | sigma {old_sigma:.5f} -> {new_sigma:.5f}')


