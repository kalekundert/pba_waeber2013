#!/usr/bin/env python3

"""
Implementation of the stochastic root-finding algorithm described by 
[Waeber2013].
"""

__version__ = '0.1.0'

import numpy as np
from numpy import sign, log, exp
from dataclasses import dataclass
from typing import Tuple

class FunctionWrapper:

    def __init__(self, f, args, kwargs, n_max):
        self.f = f
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.obs = np.zeros((2, n_max))
        self.n = 0

    def __call__(self, x):
        y = self.f(x, *self.args, **self.kwargs)
        self.obs[:,self.n] = x, y
        self.n += 1
        return y

    @property
    def x_obs(self):
        return self.obs[0,:self.n]

    @property
    def f_obs(self):
        return self.obs[1,:self.n]

class PosteriorDist:

    @classmethod
    def empty(cls, bounds, n_max):
        # The posterior distribution is represented by the *xs* and *log_ps* 
        # arrays.  The first contains the values of x that have been sampled, 
        # and the starting bounds.  The second contains the log probability 
        # density that the root falls between the corresponding value of x and 
        # the next.  Note that the first array has one more element than the 
        # second.  These arrays are allocated enough space so that they will 
        # not need to be resized at any point during the algorithm.  

        xs = np.zeros(n_max + 2)
        xs[0:2] = bounds

        log_ps = np.zeros(n_max + 1)

        return cls(xs, log_ps, 0)

    def __init__(self, xs, log_ps, n):
        self._xs = np.asfarray(xs)
        self._log_ps = np.asfarray(log_ps)
        self.n = n

        assert len(self._xs) == len(self._log_ps) + 1
        assert self._xs.ndim == self._log_ps.ndim == 1

    def median(self):
        # This expression to calculate the PDF is unstable, for two 
        # reasons:
        # 
        # - The subtraction will lose precision as the x coordinates get very 
        #   close together.
        # - The `exp()` will overflow as the log probability densities get very 
        #   large.
        # 
        # Some ways I can imagine dealing with this:
        #
        # - The subtraction is the bigger problem.  If I can deal with that, I 
        #   can avoid the overflow problem by doing: `exp(log(diff) + log_ps)`
        #
        # - I could try to somehow increase the precision when calculating the 
        #   median, but my guess is that the bisection would pretty quickly eat 
        #   up any extra precision I gave it.
        #
        # - Maybe I could store `log(Δx)` instead of just `x`.  But the problem 
        #   is calculating `Δx` in the first place, and that wouldn't help.
        # 
        # - I could set some limit on how close two x coordinates can be before 
        #   they're considered the same.

        dx = self.xs[-1] - self.xs[0]
        pdf = np.diff(self.xs) * exp(self.log_ps) / dx

        # Construct the CDF so its elements correspond to the elements of *xs*. 
        # This means setting the first position to 0 and starting the 
        # cumulative sum in the second position.

        cdf = np.zeros(self.xs.shape)
        np.cumsum(pdf, out=cdf[1:])

        x_i = np.searchsorted(cdf, 0.5)
        adj = slice(x_i - 1, x_i + 1)
        x = np.interp(0.5, cdf[adj], self.xs[adj])

        return x, x_i

    def update(self, x, x_i, sign_guess, p_c, slope):
        """
        - x and x_i must be the median.
        - x_i refers to the indices between the elements of *xs*, as in 
          slicing.
        """
        xs = self._xs
        log_ps = self._log_ps
        n = self.n

        # Ensure that the observations stay sorted:
        assert 0 < x_i < n + 2
        assert x >= xs[x_i - 1]
        assert x <= xs[x_i]

        # Make space for the new observation (if we haven't seen it before):
        if xs[x_i] != x:
            assert len(xs)     - 2 >= n + 1
            assert len(log_ps) - 1 >= n + 1

            xs[x_i+1:n+3] = xs[x_i:n+2]
            xs[x_i] = x

            log_ps[x_i:n+2] = log_ps[x_i-1:n+1]

            self.n += 1
            n = self.n

        # if sign_guess ==  1: p = p_c
        # if sign_guess == -1: p = 1 - p_c
        p = -sign(slope) * sign_guess * (p_c - 0.5) + 0.5

        # These expressions tend to overflow.  It would probably be better to 
        # store logarithms instead of raw numbers.
        log_ps[0:x_i] += log(2 * (1 - p))
        log_ps[x_i:n+1] += log(2 * p)

    @property
    def xs(self):
        return self._xs[:self.n+2]

    @property
    def log_ps(self):
        return self._log_ps[:self.n+1]

class ConfidenceIntervals:

    @classmethod
    def empty(cls, n_max):
        ci = np.zeros((4, n_max))
        return cls(ci, 0)

    def __init__(self, ci, n):
        self._ci = np.asfarray(ci)
        self.n = n

        assert self._ci.shape[0] == 4
        assert self._ci.shape[1] >= n

    def update(self, posterior, p_c, alpha):
        xs = posterior.xs
        log_ps = posterior.log_ps
        ci = self._ci
        n = self.n
        n1 = n + 1
        self.n += 1

        # These equations come from [Waeber2013] §3.3 and §3.4.
        # - d goes from 0 to `ln(2)` as `p_c` goes from 0.5 to 1.
        q_c = 1 - p_c
        d = p_c * log(2 * p_c) + q_c * log(2 * q_c)
        beta = log(p_c / q_c)

        a = n1 * d - n1**0.5 * (-0.5 * log(alpha / (n1 + 1)))**0.5 * beta
        b = n1 * d - n1**0.5 * (-0.5 * log(alpha / 2       ))**0.5 * beta

        K = np.flatnonzero(log_ps > a)
        J = np.flatnonzero(log_ps > b)

        x_lo = xs[J[ 0]]
        x_hi = xs[J[-1] + 1]

        x_lo_seq = xs[K[ 0]]
        x_hi_seq = xs[K[-1] + 1]

        if n > 0:
            x_lo_seq = max(x_lo_seq, ci[2, n-1])
            x_hi_seq = min(x_hi_seq, ci[3, n-1])

        ci[:,n] = x_lo, x_hi, x_lo_seq, x_hi_seq

    def width(self, n=-1):
        return self.ci[1,n] - self.ci[0,n]

    def width_seq(self, n=-1):
        return self.ci_seq[1,n] - self.ci_seq[0,n]

    @property
    def ci(self):
        return self._ci[0:2,:self.n]

    @property
    def ci_seq(self):
        return self._ci[2:4,:self.n]

@dataclass
class Result:
    x: float
    x_obs: np.ndarray
    f_obs: np.ndarray
    x_post: np.ndarray
    log_p_post: np.ndarray
    ci: np.ndarray
    ci_seq: np.ndarray
    converged: bool

    @property
    def p_post(self):
        return exp(self.log_p_post)

def pba_waeber2013(
        f, a, b, *,
        tol=None,
        rtol=None,
        alpha=0.05,
        p_c=0.65,
        maxiter=1000,
        check_bounds=False,
        slope=None,
        args=None,
        kwargs=None,
        #restart=None,
):
    """
    If you have a discrete domain, just interpolate.  The algorithm will still 
    work.  See [Waeber2013], Figure 4.13 and surrounding discussion.

    API inspired by scipy root finding algorithms.
    """
    f = FunctionWrapper(f, args, kwargs, maxiter)
    posterior = PosteriorDist.empty((a, b), maxiter)
    ci = ConfidenceIntervals.empty(maxiter)
    converged = False

    if check_bounds:
        sign_lo = determine_sign(f, a, p_c, maxiter - f.n)
        sign_hi = determine_sign(f, b, p_c, maxiter - f.n)
        slope = sign_hi

        if sign_lo == sign_hi:
            raise RuntimeError(f"no solution; both bounds have the same sign: {sign_lo}")

    if not slope:
        raise ValueError("no slope specified")

    while f.n < maxiter:
        x, x_i = posterior.median()
        sign_guess = determine_sign(f, x, p_c, maxiter - f.n)
        posterior.update(x, x_i, sign_guess, p_c, slope)
        ci.update(posterior, p_c, alpha)

        if ci.width() < get_tol(x, tol, rtol):
            converged = True
            break

    x, _ = posterior.median()

    return Result(
            x=x,
            x_obs=f.x_obs,
            f_obs=f.f_obs,
            x_post=posterior.xs,
            log_p_post=posterior.log_ps,
            ci=ci.ci,
            ci_seq=ci.ci_seq,
            converged=converged,
    )

def determine_sign(f, x, p_c, n_max):
    """
    Repeatedly evaluate *f(x)* until the probability that we know its sign 
    exceeds *p_c*.  Return the sign in question, or 0 if we are not able to 
    make a determination after *n_max* iterations.

    Supposedly this algorithm is described by [Siegmund1974], but the math in 
    that reference is way over my head.  I got the test of one power equation 
    from a personal communication with Rolf Waeber.
    """
    random_walk = 0
    alpha = 2 - (2 * p_c)

    for i in range(1, n_max+1):
        random_walk += sign(f(x))
        test_of_power_1 = ((i + 1) * (log(i + 1) + 2 * log(1 / alpha)))**0.5

        if abs(random_walk) > test_of_power_1:
            return sign(random_walk)

    return 0

def get_tol(x, atol, rtol):
    """
    Calculate an absolute convergence tolerance, given the optional *atol* and 
    *rtol* parameters.  If neither tolerance is specified, return 0.  This will 
    ensure that the algorithm runs until it reaches *maxiters*.
    """
    if atol:
        return atol
    if rtol:
        return abs(x) * rtol
    return 0
