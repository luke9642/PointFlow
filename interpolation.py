from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from args import get_args
from datasets import get_datasets, init_np_seed
from models.networks import PointFlow
from utils import resume, set_dir


def visualize_point_clouds(pts, pert_order=[0, 1, 2]):
    try:
        pts = pts.cpu().detach().numpy()
    except AttributeError:
        pass

    pts = pts[:, :, :, pert_order]

    fig = plt.figure(figsize=(pts.shape[1] * 10, pts.shape[0] * 10))

    k = 0

    for i, x in enumerate(pts):
        for j, y in enumerate(x):
            k += 1
            ax = fig.add_subplot(pts.shape[0], pts.shape[1], k, projection='3d')
            ax.scatter(y[:, 0], y[:, 1], y[:, 2], s=5)

    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res


def interpolate(args, model, n_interpolations=1, space=5):
    tr_dataset, _ = get_datasets(args)
    from torch.utils.data import DataLoader
    test_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
        worker_init_fn=init_np_seed)

    if n_interpolations * 2 > args.batch_size:
        take = np.math.ceil(n_interpolations * 2 / args.batch_size)
        print(f'Taking {take * args.batch_size} samples')
    else:
        take = 1
        print(f'Taking {n_interpolations * 2} samples')

    import itertools
    for batch in itertools.islice(iter(test_loader), take):
        batch = batch['test_points'].cuda()

        if take == 1:
            batch = batch[:n_interpolations * 2]

        batch = batch[torch.randperm(batch.size(0))]

        x1 = batch[0::2, :]
        x2 = batch[1::2, :]

        # y, x = model.reconstruct(batch)
        # X = visualize_point_clouds(torch.stack((batch, y, x), dim=1), [0, 2, 1])
        # imageio.imsave(Path('interpolation/images') / f'tmp1.png', X.transpose(1, 2, 0))
        # y, x, _ = model.triang_recon(batch, depth=4)
        # X = visualize_point_clouds(torch.stack((y, x), dim=1), [0, 2, 1])
        # imageio.imsave(Path('interpolation/images') / f'tmp2.png', X.transpose(1, 2, 0))
        # X = visualize_point_clouds(batch[None, :], [0, 2, 1])
        # imageio.imsave(Path('interpolation/images') / f'tmp3.png', X.transpose(1, 2, 0))
        # raise SystemExit

        print('interpolating...')
        if args.triangulation:
            yield model.triang_interpolate(x1, x2, space, args.method, args.depth).cpu()
        else:
            yield model.interpolate(x1, x2, space).cpu()


def main(args):
    save_path = set_dir(Path('interpolation') / args.log_name)
    images_path = set_dir(save_path / 'images')

    model = PointFlow(args)
    model = model.cuda()
    model.multi_gpu_wrapper(nn.DataParallel)
    print(f'Resume Path: {args.resume_checkpoint}')
    model, _, start_epoch = resume(
        args.resume_checkpoint, model, optimizer=None, strict=(not args.resume_non_strict))

    if args.overwrite:
        model.sphere.m = args.m
        model.sphere.sigma = args.sigma

    model.eval()

    x_int = torch.cat(list(interpolate(args, model, args.n_interpolations, args.space)), dim=1).transpose(1, 0)
    print('Interpolated shape:', x_int.shape)

    result_path = f'inter_mesh_depth-{args.depth}_epoch-{start_epoch}' if args.triangulation else f'inter_epoch-{start_epoch}'
    torch.save(x_int, save_path / f'{result_path}.pt')

    X = visualize_point_clouds(x_int, [0, 2, 1])
    imageio.imsave(images_path / f'{result_path}.png', X.transpose(1, 2, 0))

    print('Results saved to', save_path / f'{result_path}.pt')


if __name__ == '__main__':
    args = get_args()
    main(args)
