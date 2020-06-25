import argparse
from pathlib import Path

import numpy as np
import imageio
import torch
import matplotlib.pyplot as plt

from utils import visualize_point_clouds


def set_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_tensor(x):
    return torch.load(x, map_location=torch.device('cpu')) if x else None


def visualize_point_cloud(pts, pert_order=[0, 1, 2]):
    try:
        pts = pts.cpu().detach().numpy()
    except AttributeError:
        pass

    pts = pts[:, pert_order]

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5)

    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res


def get_args():
    parser = argparse.ArgumentParser(description='Open mlab mesh and save it to file')
    parser.add_argument('samples', type=load_tensor, help='PyTorch tensor with sample data', metavar='samples_or_recons')
    parser.add_argument('--spheres', type=load_tensor, help='PyTorch tensor with sample data', metavar='spheres_or_truths', default=None)
    parser.add_argument('--output_path', default='triang/mesh', type=set_dir, help='Directory, where pictures are saved')
    parser.add_argument('--sample_num', default=0, type=int, help='Sample num to visualize')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if args.spheres:
        X = np.concatenate([visualize_point_clouds(sample, sphere, idx, [0, 2, 1])
                            for idx, (sample, sphere) in enumerate(zip(args.samples, args.spheres))], axis=1)
    else:
        shapes = [visualize_point_cloud(sample, [0, 2, 1]) for sample in args.samples.permute(2, 0, 1)]
        print(len(shapes))

        X1 = np.concatenate(shapes[:len(shapes) // 2], axis=2)
        X2 = np.concatenate(shapes[len(shapes) // 2:], axis=2)
        X = np.concatenate([X1, X2], axis=1)

    imageio.imsave(args.output_path / 'samples.png', X.transpose(1, 2, 0))
