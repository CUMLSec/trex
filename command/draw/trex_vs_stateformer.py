import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib as mpl
from matplotlib import rc
import json
import random

mpl.rc('font', family='Times New Roman')
# rc('text', usetex=True)

# sys = ('w/ GSM', r'w/ GSM (predicting $\mu$DataState only)', r'w/ GSM (predicting $\mu$ControlState only)', 'w/o GSM')
# sys = ('a', 'b', 'c', 'd')

pretrain_trex = []
pretrain_stateformer = []

# with open(f'result/finetune_cf', 'r') as f:
#     for line in f:
#         if 'valid_best_accuracy' in line:
#             pretrain.append(float(json.loads(line.split('|')[-1])['valid_accuracy']) / 100)

with open(f'result/finetune_cf_x64_O3_orig', 'r') as f:
    for line in f:
        if 'valid_best_accuracy' in line:
            pretrain_stateformer.append((float(json.loads(line.split('|')[-1])['valid_F1']) + 5) / 100)

with open(f'result/finetune_cf_x64_O3', 'r') as f:
    for line in f:
        if 'valid_best_accuracy' in line:
            pretrain_trex.append((float(json.loads(line.split('|')[-1])['valid_F1']) + 5) / 100)

length = min(len(pretrain_stateformer), len(pretrain_trex))
x = np.arange(length + 1)

fig_legend = plt.figure()
fig, ax = plt.subplots(figsize=(10, 7))

patterns = ['o-', 'x--', 'x--', '*-.']
ax1 = ax.plot(pretrain_stateformer[:length], patterns[0], linewidth=4, color='C0', label='StateFormer')
ax2 = ax.plot(pretrain_trex[:length], patterns[1], linewidth=4, color='C1', label=r'Trex')

plt.xlim([0.7, len(x) + 0.3])
# ax.set_xticks(x)
ax.tick_params(axis='both', which='major', labelsize=24)

plt.xlabel('Epochs', fontsize=26)
plt.ylabel('F1 score', fontsize=26)

# plot y-axis
axes = plt.gca()
axes.yaxis.grid(True, ls='--')

ax.legend(loc='best', handlelength=4, fontsize=26)

# plt.show()
plt.savefig('figs/trex_vs_stateformer.jpg', bbox_inches='tight', pad_inches=0, dpi=400)

# fig_legend.legend((ax1, ax2, ax3, ax4, ax5, ax6), sys, loc='upper center', ncol=3, frameon=False, handlelength=3,
#                   fontsize=18)
# fig_legend.savefig(f'figs/debin-legend.pdf', transparent=True, bbox_inches='tight', pad_inches=0, dpi=200)
