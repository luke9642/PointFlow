import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mesh.utils import PointCloud, Mesh
from metrics.sphere_triangles import generate
from utils import load_tensor, set_dir


def grid_arrays(arrays):
    return np.vstack([np.hstack(arrays_row) for arrays_row in arrays])


def get_points(samples, scale=1, scale_rows=()):
    return np.stack([[PointCloud(sample, scale=scale if i in scale_rows or not scale_rows else 1, active_color=i * 2, order=(0, 2, 1)).to_numpy() for sample in samples_row] for i, samples_row in enumerate(samples)])


def get_meshes(samples, triangles, scale=1, scale_rows=()):
    return np.stack([[Mesh(sample, triangles, scale=scale if i in scale_rows or not scale_rows else 1, active_color=i, order=(0, 2, 1)).to_numpy() for sample in samples_row] for i, samples_row in enumerate(samples)])


def get_args():
    parser = argparse.ArgumentParser(description='Open mlab mesh and save it to file')
    parser.add_argument('samples', type=load_tensor, nargs='+', help='PyTorch tensor with sample data', metavar='samples_or_recons')
    parser.add_argument('-m', '--mode', choices=('points', 'mesh', 'both'), default='both')
    parser.add_argument('-t', '--dtype', choices=('recon', 'sample', 'interp'), default='sample')
    parser.add_argument('--method', default='edge', help='')
    parser.add_argument('--depth', type=int, default=4, help='')
    parser.add_argument('--scale', type=float, default=1, help='')
    parser.add_argument('-r', '--scale_rows', type=int, default=(), nargs='*', help='')
    # parser.add_argument('--output_path', default='triang/mesh', type=set_dir, help='Directory, where pictures are saved')
    parser.add_argument('-s', '--samples_to_save', type=int, nargs='*', help='Samples nums to save')
    parser.add_argument('--permute', type=int, nargs='*', help='')
    parser.add_argument('--shuffle', action='store_true', help='')
    return parser.parse_args()


def generate_images(samples, mode, results_path, scale=1, scale_rows=(), method='edge', depth=4):
    arrays = None
    triangs_path = set_dir(results_path / 'triangs')

    if mode == 'points':
        arrays = get_points(samples, scale, scale_rows)
    elif mode == 'mesh':
        arrays = get_meshes(samples, get_triangulation(method, depth, triangs_path), scale, scale_rows)
    elif mode == 'both':
        points = get_points(samples, scale)
        mesh = get_meshes(samples, get_triangulation(method, depth, triangs_path), scale, scale_rows)

        assert len(points.shape) == 5, 'Shape should be: (rows, columns, resolution, 3)'
        assert len(mesh.shape) == 5, 'Shape should be: (rows, columns, resolution, 3)'
        return points, mesh

    return arrays


def sample(samples, mode, results_path, scale=1, scale_rows=()):  # TODO reshaping
    if len(samples.shape) == 3:
        samples = samples[:, None, :]
        print('Updated shape:', samples.shape)

    if mode == 'both':
        points, mesh = generate_images(samples, 'both', results_path, scale, scale_rows)
        arrays = np.concatenate((points, mesh), axis=1)
    else:
        arrays = generate_images(samples, mode, results_path, scale, scale_rows)
    return arrays


def reconstruct(recon, truth, mode, results_path, scale=1, scale_rows=()):
    if len(recon.shape) == 3:
        recon = recon[:, None, :]
        print('Updated recon shape:', recon.shape)
    if len(truth.shape) == 3:
        truth = truth[:, None, :]
        print('Updated truth shape:', truth.shape)

    if mode == 'points':
        truth = generate_images(truth, 'points', results_path, scale, scale_rows)
        points = generate_images(recon, 'points', results_path, scale, scale_rows)
        arrays = np.vstack((truth, points))
    elif mode == 'both':
        truth = generate_images(truth, 'points', results_path, scale, scale_rows)
        points, mesh = generate_images(recon, 'both', results_path, scale, scale_rows)
        print('truth', truth.shape)
        print('points', points.shape)
        print('mesh', mesh.shape)
        arrays = np.hstack((truth, points, mesh))
    else:
        raise RuntimeError('Not supported mode for reconstruction')
    return arrays


def interpolate(samples, mode, results_path, scale=1, scale_rows=()):
    if mode == 'both':
        points, mesh = generate_images(samples, 'both', results_path, scale, scale_rows)
        arrays = np.stack((points, mesh))
        arrays = arrays.swapaxes(0, 1).reshape(-1, *arrays.shape[2:])
    else:
        arrays = generate_images(samples, mode, results_path, scale, scale_rows)
    return arrays


def shuffle(samples: np.ndarray):
    shape = samples.shape
    samples = samples.reshape((-1, *shape[2:]))
    np.random.shuffle(samples)
    samples = samples.reshape(shape)
    return samples


def main(args):
    arrays = None
    results_path = set_dir('results')

    subset = args.samples_to_save or slice(None)
    args.samples = [s[subset] for s in args.samples]

    if args.samples_to_save:
        return save_samples(args.samples, subset, args.scale, args.mode, args.dtype, results_path)

    if args.dtype == 'recon':
        assert len(args.samples) == 2
        recon, truth = args.samples
        arrays = reconstruct(recon, truth, args.mode, results_path, args.scale, args.scale_rows)
    elif args.dtype == 'sample':
        assert len(args.samples) == 1
        samples = args.samples[0]
        if args.shuffle:
            samples = shuffle(samples)
        arrays = sample(samples, args.mode, results_path, args.scale, args.scale_rows)
    elif args.dtype == 'interp':
        assert len(args.samples) == 1
        arrays = interpolate(args.samples[0], args.mode, results_path, args.scale, args.scale_rows)

    assert len(arrays.shape) == 5, f'Incorrect results shape: {arrays.shape}'
    params = '' if args.mode == 'points' else f'-method-{args.method}-depth-{args.depth}'
    results_path = results_path / f'grid-{args.dtype}-{args.mode}{params}.png'
    print('Results shape:', arrays.shape)
    images_grid = grid_arrays(arrays.reshape((arrays.shape[0] // 2, arrays.shape[1] * 2, *arrays.shape[2:])))
    Image.fromarray(images_grid).save(results_path)
    print('Saved results to:', results_path)


# def main(args):
    # mlab.options.offscreen = True

    # args.samples = preprocess(args.samples)
    #
    # print('Input shape:', args.samples.shape)
    #
    # if args.permute:
    #     args.samples = args.samples.permute(*args.permute)
    #     print('Permuted shape:', args.samples.shape)
    #
    # if len(args.samples.shape) == 3:
    #     args.samples = args.samples[:, None, :]
    #     print('Updated shape:', args.samples.shape)
    #
    # results_path = set_dir('results')
    # triangs_path = set_dir(results_path / 'triangs')

    # if args.samples_to_save:
    #     save_samples(args.samples, args.samples_to_save, args.scale, args.mode, results_path)
    #     return

    # if args.mode == 'points':
    #     arrays = get_points(args.samples, scale=args.scale)
    #     results_path = results_path / f'grid-points.png'
    # elif args.mode == 'mesh':
    #     triangles = get_triangulation(args.method, args.depth, triangs_path)
    #     arrays = get_meshes(args.samples, triangles, scale=args.scale)
    #     results_path = results_path / f'grid-mesh-method-{args.method}-depth-{args.depth}.png'
    # else:
        # triangles = get_triangulation(args.method, args.depth, triangs_path)
        # points = get_points(args.samples, scale=args.scale)
        # mesh = get_meshes(args.samples, triangles, scale=args.scale)
        # assert len(points.shape) == 5, 'Shape should be: (rows, columns, resolution, 3)'
        # assert len(mesh.shape) == 5
        #
        # if args.vertical_both:
        #     arrays = np.concatenate((points, mesh), axis=1)
        # else:
        #     arrays = np.stack((points, mesh))
        #     arrays = arrays.swapaxes(0, 1).reshape(-1, *arrays.shape[2:])
        # print('=' * 5, arrays.shape)
        # results_path = results_path / f'grid-both.png'
    # assert len(arrays.shape) == 5
    #
    # images_grid = grid_arrays(arrays)
    # Image.fromarray(images_grid).save(results_path)
    # print('Saved results to:', results_path)


def save_samples(samples, samples_to_save, scale, mode, dtype, results_path):
    save_path = set_dir(results_path / 'saved')
    import torch
    name = f'{dtype}_{mode}_{len(samples_to_save)}_samples'

    samples_path = (save_path / name).with_suffix('.pt')
    i = 0
    while samples_path.exists():
        i += 1
        samples_path = (save_path / f'{name}[{i}]').with_suffix('.pt')

    torch.save(samples, samples_path)
    arrays = get_points(samples, scale=scale)
    images_grid = grid_arrays(arrays)
    Image.fromarray(images_grid).save(samples_path.with_suffix('.png'))
    print(f'Saved {samples_to_save} to {samples_path}')


def get_triangulation(method, depth, triangs_path):
    triang_path = Path(triangs_path / f'triang-method_{method}-depth_{depth}.pt')
    if triang_path.exists():
        print('Loaded triang')
        triangulation = load_tensor(triang_path)
    else:
        _, T = generate('edge', depth)
        triangulation = T.triangles
        torch.save(triangulation, str(triang_path))
        print(f'Saved triang to {triangs_path}')
    return triangulation


if __name__ == '__main__':
    main(get_args())

