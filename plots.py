#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# pd.options.display.width = 0
pd.set_option('display.expand_frame_repr', False)

# %%
with open('tmp/metrics.log') as f:
    lines = [line.replace('[ ', '[_').replace(' ]', '_]').split(' ') for line in f.readlines()]

#%%
df = pd.DataFrame([dict(zip(line[::2], line[1::2])) for line in lines]).drop(columns='[Rank')

df = df.astype({'Epoch': int, 'Entropy': float, 'LatentNats': float, 'PointNats': float, 'LogLikelihood': float})

df

#%%
# df = -df.groupby('Epoch', as_index=True)['LogLikelihood'].agg(['min', 'mean', 'max'])
df['LogLikelihood'] = -df['LogLikelihood']
df

#%%
df = df.pivot_table(values='LogLikelihood', index='Epoch')
df

#%%
sns.set_style('ticks')
sns.set_context('paper')
ax = sns.relplot(x='Epoch', y='LogLikelihood', kind='line', data=df, ci="sd")
# ax = sns.lineplot(data=df, dashes=False)
plt.show()

#%%
ax.savefig('loglikelihood.png')

# %%
df = (pd.read_csv('checkpoints/ae/shapenet15k-airplane-gauss-most-tiny-m-sigma-schedule/log_likelihood.csv')
      .drop(columns=['Unnamed: 0', 'bidx'])
      .rename(columns={'value': 'log_likelihood'}))

df['prior_nats'] = df['prior_nats'].str.slice(7, 15)
df['recon_nats'] = df['recon_nats'].str.slice(7, 13)

df

# %%
df = df.astype({'epoch': int, 'entropy': float, 'prior_nats': float, 'recon_nats': float, 'log_likelihood': float})
df

# %%
df = df.pivot_table(values=['prior_nats', 'recon_nats', 'log_likelihood'], index='epoch')
df

# %%
df['log_likelihood'] = -df['log_likelihood']

# %%
max_val = 10_000
df['log_likelihood'] = df['log_likelihood'][df['log_likelihood'] < max_val]

# %%
from matplotlib import rcParams

# figure size in inches
# rcParams['figure.figsize'] = 11.7,8.27
sns.set(rc={'figure.figsize': (5, 10)})
sns.set_style('ticks')
sns.set_context('paper')
fig = plt.figure()
ax = fig.add_subplot(3, 1, 1)

sns.lineplot(x='epoch', y='log_likelihood', data=df, ci="sd", ax=ax)
with sns.color_palette("YlGn", 1):
    ax = fig.add_subplot(3, 1, 2)
sns.lineplot(x='epoch', y='prior_nats', data=df, ci="sd", ax=ax)
with sns.color_palette('YlGnBu', 1):
    ax = fig.add_subplot(3, 1, 3)
    ax.set(yscale="log")
sns.lineplot(x='epoch', y='recon_nats', data=df, ci="sd", ax=ax)

# ax = sns.lineplot(data=df, dashes=False)
plt.savefig('loglikelihood.png', dpi=200)

plt.show()

# %%

# %%

