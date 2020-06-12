import argparse
from pathlib import Path

import numpy as np
import imageio
import torch
from mayavi import mlab

from utils import visualize_point_clouds


def set_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_tensor(x):
    return torch.load(x, map_location=torch.device('cpu'))


def visualize_mesh(XX, T, path, order=(0, 1, 2)):
    (x, y, z) = order
    f = mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1), size=(1000, 1000))
    mlab.view(azimuth=160, elevation=90, figure=f)
    mlab.triangular_mesh(XX[:, x], XX[:, y], XX[:, z], T, transparent=True, representation='wireframe', color=(0, 0, 0))

    mlab.triangular_mesh(XX[:, x], XX[:, y], XX[:, z], T)
    mlab.savefig(str(path))
    mlab.show()


def get_args():
    parser = argparse.ArgumentParser(description='Open mlab mesh and save it to file')
    parser.add_argument('samples', type=load_tensor, help='PyTorch tensor with sample data', metavar='samples_or_recons')
    parser.add_argument('spheres', type=load_tensor, help='PyTorch tensor with sample data', metavar='spheres_or_truths')
    parser.add_argument('triangulation', type=load_tensor, help='Numpy array with triangulation data')
    parser.add_argument('--output_path', default='triang/mesh', type=set_dir, help='Directory, where pictures are saved')
    parser.add_argument('--sample_num', default=0, type=int, help='Sample num to visualize')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    print(f'''\
Samples shape {tuple(args.samples.shape)}
Spheres shape {tuple(args.spheres.shape)}
Triang shape {args.triangulation.shape}
Triang max value {args.triangulation.max()}\
    ''')

    X = np.concatenate([visualize_point_clouds(sample, sphere, idx)
                        for idx, (sample, sphere) in enumerate(zip(args.samples, args.spheres))], axis=1)
    imageio.imsave(args.output_path / 'samples.png', X.transpose(1, 2, 0))
    visualize_mesh(args.samples.detach().numpy()[args.sample_num], args.triangulation, args.output_path / 'mesh.png', order=(1, 2, 0))
