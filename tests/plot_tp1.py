#!/usr/bin/env python3

import numpy as np
from numpy import log
from matplotlib.pyplot import *

p_cs = [0.6, 0.7, 0.8, 0.9]
random_walks = [
    '20 * [1]',
    '[-1] + 19 * [1]',
    '5 * [-1, -1, 1, 1]',
]
R, C = len(p_cs), len(random_walks)

fig, axes = subplots(R, C, figsize=(3*C, 3*R), sharex=True, sharey=True)
n = np.arange(20) + 1

for i, p_c in enumerate(p_cs):
    for j, rw in enumerate(random_walks):
        alpha = 2 - 2 * p_c
        tp1 = lambda n: ((n + 1) * (log(n + 1) + 2 * log(1 / alpha)))**0.5

        axes[i,j].plot(n, tp1(n))
        axes[i,j].plot(n, np.abs(np.cumsum(eval(rw))))

for ax in axes.flat:
    ax.set_aspect('equal')
for ax, p_c in zip(axes[:,0], p_cs):
    ax.set_ylabel(f'{p_c=}\nn')
for ax, rw in zip(axes[0,:], random_walks):
    ax.set_title(rw)
for ax in axes[-1,:]:
    ax.set_xlabel('n')

tight_layout()
show()



