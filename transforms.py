from pathlib import Path
import torch
from PIL import Image, ImageDraw
import numpy as np
from results import interpolate, grid_arrays

# %%

samples = [torch.load(file) for file in Path('results/saved/samples/points').iterdir() if file.suffix == '.pt']
t = torch.cat(samples)
res = t[:20].reshape(4, 5, *t.shape[2:])
torch.save(res, 'results/saved/sample.pt')

# %%

samples = [torch.load(file) for file in Path('results/saved/reconstructions/points').iterdir() if file.suffix == '.pt']

for s in samples:
    print(s.shape)
# %%
t = torch.cat(samples)
t.shape
# %%
res = t[:30].reshape(15, 4, *t.shape[2:])
torch.save(res, 'results/saved/recon.pt')

# %%

samples = [torch.load(file) for file in Path('results/saved/interpolation/points').iterdir() if file.suffix == '.pt']

for s in samples:
    print(s.shape)

# %%
# j = 0
def tmp(s, scale=1.):
    # global j
    # j += 1
    return interpolate(s, 'points', Path('results'), scale=scale)


# arrays = [tmp(s, 1.8) if j in (0, 1) else tmp(s) for s in samples]
arrays = [tmp(s) for s in samples]

res = np.concatenate(arrays)
res.shape

# %%
img = Image.fromarray(grid_arrays(res))
img.save('results/saved/interp.png')

# %%
# res = t[:30].reshape(15, 4, *t.shape[2:])
torch.save(res[:1], 'results/saved/recon.pt')

# %%
img: Image.Image = Image.open('results/grid-recon-both-method-edge-depth-4.png')
draw = ImageDraw.Draw(img)
draw.line((img.width//2, 0, img.width//2, img.height), fill=15, width=3)
img.save(r'C:\Users\Luke\Desktop\grid-recon-both.png')

# %%
samples = [torch.load(file) for file in Path('results/saved/samples/both').iterdir() if file.suffix == '.pt']
t = torch.cat(samples)
res = t[:20].reshape(4, 5, *t.shape[2:])
torch.save(res, 'results/saved/sample.pt')

# %%
samples = [torch.load(file) for file in Path('results/saved/reconstructions/both').iterdir() if file.suffix == '.pt']

recon = []
truth = []
for s in samples:
    r, t = s
    recon.append(r)
    truth.append(t)
# %%
r = torch.cat(recon)
t = torch.cat(truth)
r.shape, t.shape

# %%
# rres = r[:24].reshape(6, 4, *r.shape[1:])
# tres = t[:24].reshape(6, 4, *t.shape[1:])
torch.save(r, 'results/saved/recon.pt')
torch.save(t, 'results/saved/truth.pt')

# %%
samples = [torch.load(file) for file in Path('results/saved/interpolation/both').iterdir() if file.suffix == '.pt']

for s in samples:
    print(s.shape)

# %%
def tmp(s, scale=1.):
    # global j
    # j += 1
    return interpolate(s, 'both', Path('results'), scale=scale)


# arrays = [tmp(s, 1.8) if j in (0, 1) else tmp(s) for s in samples]
arrays = [tmp(s) for s in samples]
# %%
for s in arrays:
    print(s.shape)

# %%
res = np.concatenate(arrays)
res.shape

# %%
img = Image.fromarray(grid_arrays(res))
img.save('results/saved/interp.png')

# %%
img: Image.Image = Image.open('results/saved/interp.png')
draw = ImageDraw.Draw(img)
for i in range(1, 9):
    draw.line((0, img.height // 9 * i, img.width, img.height // 9 * i), fill=15, width=3)
img.save(r'C:\Users\Luke\Desktop\grid-recon-both.png')
