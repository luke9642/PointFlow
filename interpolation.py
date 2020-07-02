from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from args import get_args
from datasets import get_datasets, init_np_seed
from models.networks import PointFlow
from utils import resume, save_image


def set_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_tensor(x):
    return torch.load(x, map_location=torch.device('cpu'))


def visualize_point_clouds(pts, pert_order=[0, 1, 2]):
    try:
        pts = pts.cpu().detach().numpy()
    except AttributeError:
        pass

    pts = pts[:, :, :, pert_order]

    fig = plt.figure(figsize=(pts.shape[1] * 3, pts.shape[0] * 3))

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


def decode(model, alpha, z1, z2, i, size):
    print(f'{i} / {size}')
    return model.decode((1 - alpha) * z1 + alpha * z2, z1.size(1))[1]


def interpolate(args, model, n_interpolations=1, space=5):
    _, te_dataset = get_datasets(args)
    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=n_interpolations * 2, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
        worker_init_fn=init_np_seed)

    batch = next(iter(test_loader))['test_points']

    x1 = batch[0::2, :]
    x2 = batch[1::2, :]

    print('interpolating...')
    return model.triang_interpolate(x1, x2, space)
    # return model.interpolate(x1, x2, space)

    # x_interpolated = [decode(model, alpha, z1, z2, i, space) for i, alpha in enumerate(np.linspace(0, 1, space), start=1)]
    # print('...done')
    # return torch.stack(x_interpolated, dim=0)

    # for step, x in enumerate(test_loader):
    #     if step > n_interpolations:
    #         break
    #
    #     x1 = x['test_points'][None, 0, :]
    #     x2 = x['test_points'][None, 1, :]
    #
    #     z1 = model.encode(x1)
    #     z2 = model.encode(x2)
    #
    #     x_interpolated = [model.decode((1 - alpha) * z1 + alpha * z2, z1.size(1))[1] for alpha in np.linspace(0, 1, 10)]
    #     return torch.stack(x_interpolated, dim=0)


def main(args):
    save_path = Path('interpolation')
    output_path = Path('interpolation/images')
    save_path.mkdir(parents=True, exist_ok=True)

    model = PointFlow(args)

    model = model.cuda()
    model.multi_gpu_wrapper(nn.DataParallel)
    print(f'Resume Path: {args.resume_checkpoint}')
    model, _, start_epoch = resume(
        args.resume_checkpoint, model, optimizer=None, strict=(not args.resume_non_strict))

    model.eval()

    X, T = interpolate(args, model, 10, 10).transpose(1, 0)
    print(f'T: {T.shape}')
    torch.save(X.permute(1, 2, 0), save_path / 'inter_mesh.pt')

    X = visualize_point_clouds(X, [0, 2, 1])

    imageio.imsave(output_path / 'interpolated.png', X.transpose(1, 2, 0))

    # save_image(X, X, [0, 1, 2],
    #            data_path=save_path / f'interpolated-epoch_{start_epoch}.pt',
    #            image_path=output_path / 'interpolated.png')


if __name__ == '__main__':
    args = get_args()
    main(args)
