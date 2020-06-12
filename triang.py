from pathlib import Path

from args import get_args
from datasets import get_datasets, init_np_seed
from models.networks import PointFlow
import torch
import torch.nn as nn

from utils import resume
from metrics.sphere_triangles import generate


def reconstruct(args, model):
    _, te_dataset = get_datasets(args)
    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
        worker_init_fn=init_np_seed)

    truth = next(iter(test_loader))['test_points']
    z, x, T = model.triang_recon(truth)
    return truth[None, :], x[None, :], T


def main(args):
    save_path = Path('triang')
    save_path.mkdir(parents=True, exist_ok=True)

    model = PointFlow(args)

    def _transform_(m):
        return nn.DataParallel(m)

    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)
    print(f'Resume Path: {args.resume_checkpoint}')
    model, _, start_epoch = resume(
        args.resume_checkpoint, model, optimizer=None, strict=(not args.resume_non_strict))

    model.eval()

    truth, recons, triang = reconstruct(args, model)
    torch.save(truth, save_path / f'truth-epoch_{start_epoch}.pt')
    torch.save(recons, save_path / f'recons-epoch_{start_epoch}.pt')
    torch.save(triang, save_path / f'recons-triang.pt')

    spheres = torch.stack([generate(args.method, args.depth)[0] for _ in range(args.samples_num)])
    _, samples = model.triangulate(spheres)

    torch.save(spheres, save_path / f'spheres-epoch_{start_epoch}-depth_{args.depth}.pt')
    torch.save(samples, save_path / f'samples-epoch_{start_epoch}-depth_{args.depth}.pt')

    if args.save_triangulation:
        _, triang = generate(args.method, args.depth)
        torch.save(triang.triangles, save_path / f'triangulation-method_{args.method}-depth_{args.depth}.pt')


if __name__ == '__main__':
    args = get_args()
    main(args)
