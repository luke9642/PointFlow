from pathlib import Path

import imageio

import interpolation
from args import get_args
from datasets import get_datasets, init_np_seed
from models.networks import PointFlow
import torch
import torch.nn as nn

from utils import resume, set_dir
import matplotlib.pyplot as plt
import numpy as np


def visualize_point_clouds(pts, gtr, pert_order=[0, 1, 2]):
    try:
        pts = pts.cpu().detach().numpy()
        gtr = gtr.cpu().detach().numpy()
    except AttributeError:
        pass

    pts = pts[:, :, pert_order]
    gtr = gtr[:, :, pert_order]

    fig = plt.figure(figsize=(2 * 10, pts.shape[0] * 10))

    k = 0

    for x, y in zip(pts, gtr):
        k += 1
        ax = fig.add_subplot(pts.shape[0], 2, k, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=5)

        k += 1
        ax = fig.add_subplot(pts.shape[0], 2, k, projection='3d')
        ax.scatter(y[:, 0], y[:, 1], y[:, 2], s=5)

    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    plt.close()
    return res


def reconstruct(args, model):
    tr_dataset, _ = get_datasets(args)
    test_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
        worker_init_fn=init_np_seed)

    truth = next(iter(test_loader))['test_points']
    truth = truth.cuda()
    z, x = model.reconstruct(truth)
    return truth, x


def main(args):
    save_path = set_dir(Path('sample') / args.log_name)
    images_path = set_dir(save_path / 'images')

    model = PointFlow(args)
    model = model.cuda()
    model.multi_gpu_wrapper(nn.DataParallel)
    print(f'Resume Path: {args.resume_checkpoint}')
    model, _, start_epoch = resume(
        args.resume_checkpoint, model, optimizer=None, strict=(not args.resume_non_strict))

    model.eval()

    truth, recons = reconstruct(args, model)

    file_path = f'{{}}-epoch_{start_epoch}-samples_{args.batch_size}.pt'

    torch.save(truth, save_path / file_path.format('truth'))
    torch.save(recons, save_path / file_path.format('recons'))

    X = visualize_point_clouds(recons, truth, [0, 2, 1])
    imageio.imsave((images_path / file_path.format('recons')).with_suffix('.png'), X)

    _, samples = model.sample(args.batch_size, args.num_sample_points)
    torch.save(samples, save_path / file_path.format('samples'))
    X = interpolation.visualize_point_clouds(samples[:, None, :], [0, 2, 1])
    imageio.imsave((images_path / file_path.format('samples')).with_suffix('.png'), X.transpose((1, 2, 0)))
    print('Saved results to:', save_path / file_path)


if __name__ == '__main__':
    main(get_args())
