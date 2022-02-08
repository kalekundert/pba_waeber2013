#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pba_waeber2013 import pba_waeber2013

def f1(x):
    return x

def f2(x):
    return x - 1

f1.a = -1/3
f1.b = 2/3
f1.kwargs = {
        'tol': 1e-2,
}

f2.a = 2/3
f2.b = 5/3
f2.kwargs = {
        'rtol': 1e-2,
}

noise = 0.001

fig, axes = plt.subplots(2, 3)

for i, f in enumerate([f1, f2]):

    rng = np.random.default_rng(0)
    def g(x):
        return f(x) + rng.normal(0, noise)

    result = pba_waeber2013(
            g, f.a, f.b,
            p_c=0.95,
            slope=1,
            maxiter=100_000,
            **f.kwargs,
    )

    axes[i,0].plot(result.x_obs, result.f_obs, ',')
    axes[i,0].set_xlim(f.a, f.b)

    n = np.arange(result.ci.shape[1])
    axes[i,1].plot(n, result.ci[0])
    axes[i,1].plot(n, result.ci[1])

    x = np.diff(result.x_post) / 2 + result.x_post[:-1]
    axes[i,2].plot(x, result.p_post)
    axes[i,2].text(
            0.95, 0.95,
            f'x={result.x:.2f}',
            verticalalignment='top',
            horizontalalignment='right',
            transform=axes[i,2].transAxes,
    )

plt.tight_layout()
plt.show()

