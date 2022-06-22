#%%
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('ggplot')
# %%
data = pd.read_csv('benchmark.csv')
# Divide by number of runs within single line
data.loc[data['gradtype'] == 'vector', 'time'] /= 1000
data.loc[data['gradtype'] == 'full', 'time'] /= 100
data['time'] *= 1000  # s to ms
avg_data = data.groupby(['nettype', 'gradtype', 'depth', 'layer_size', 'batch_size']).median().reset_index()
# %%
depth_line = avg_data.query('layer_size == 10 & batch_size == 20')
layer_size_line = avg_data.query('depth == 5 & batch_size == 20')
batch_size_line = avg_data.query('depth == 5 & layer_size == 10')
# %%


def add_normalized_target(d, target, unit):
    d = d.copy()
    d[f'{target} (normalized)'] = d[target] / d[target].min()
    d[f'{target} ({unit})'] = d[target]
    return d


def split_by_type(d, target, unit):
    dual = d.query('nettype == "dual"')
    auto = d.query('nettype == "auto"')
    dual_vec = add_normalized_target(dual.query('gradtype == "vector"'), target, unit)
    dual_full = add_normalized_target(dual.query('gradtype == "full"'), target, unit)
    auto_vec = add_normalized_target(auto.query('gradtype == "vector"'), target, unit)
    auto_full = add_normalized_target(auto.query('gradtype == "full"'), target, unit)
    return dual_vec, dual_full, auto_vec, auto_full


# %%
dual_vec_depth, dual_full_depth, auto_vec_depth, auto_full_depth = split_by_type(depth_line, 'time', 'ms')
# %%
sns.lineplot(data=dual_vec_depth, x='depth', y='time (ms)', label='Dual')
sns.lineplot(data=auto_vec_depth, x='depth', y='time (ms)', label='Auto')
plt.title('Gradient-vector product')
plt.legend()
plt.show()
# %%
sns.lineplot(data=dual_full_depth, x='depth', y='time (ms)', label='Dual')
sns.lineplot(data=auto_full_depth, x='depth', y='time (ms)', label='Auto')
plt.title('Full gradient data')
plt.legend()
plt.show()
# %%
sns.lineplot(data=dual_vec_depth, x='depth', y='time (normalized)', label='Forward (dualgrad)')
sns.lineplot(data=auto_vec_depth, x='depth', y='time (normalized)', label='Backward (autograd)')
plt.title('Gradient-vector product')
plt.legend()
plt.show()
# %%
sns.lineplot(data=dual_full_depth, x='depth', y='time (normalized)', label='Forward (dualgrad)')
sns.lineplot(data=auto_full_depth, x='depth', y='time (normalized)', label='Backward (autograd)')
plt.title('Full gradient data')
plt.legend()
plt.show()


# %%
memory = pd.read_csv('benchmark_memory.csv')
dual_vec_mem, dual_full_mem, auto_vec_mem, auto_full_mem = split_by_type(memory, 'memory', 'MiB')
