import argparse
import pathlib

import imageio
import torch
from mayavi import mlab

from utils import visualize_point_clouds


def visualize_mesh(XX, T, path):
    f = mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1), size=(1000, 1000))
    mlab.view(azimuth=160, elevation=90, figure=f)
    mlab.triangular_mesh(XX[:, 0], XX[:, 1], XX[:, 2], T, transparent=True, representation='wireframe', color=(0, 0, 0))

    mlab.triangular_mesh(XX[:, 0], XX[:, 1], XX[:, 2], T)
    mlab.savefig(str(path))
    mlab.show()


def get_args():
    parser = argparse.ArgumentParser(description='Open mlab mesh and save it to file')
    parser.add_argument('input', help='PyTorch tensor with sample data')
    parser.add_argument('triangulation', help='PyTorch tensor with triangulation data')
    parser.add_argument('output_path', type=lambda x: pathlib.Path(x), help='Directory, where pictures are saved')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    sample = torch.load(args.input, map_location=torch.device('cpu'))
    triang = torch.load(args.triangulation, map_location=torch.device('cpu'))

    X = visualize_point_clouds(sample, sample, 0)
    imageio.imsave(args.output_path / 'sample.png', X.transpose(1, 2, 0))

    visualize_mesh(sample.detach().numpy(), triang, args.output_path / 'mesh.png')
