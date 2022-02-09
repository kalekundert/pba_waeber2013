#!/usr/bin/env python3

import pytest
import parametrize_from_file as pff
import random
import numpy as np

from pba_waeber2013 import *
from pytest import approx, raises
from voluptuous import Schema, Coerce, Optional

with_py = pff.voluptuous.Namespace('import random')
with_approx = pff.voluptuous.Namespace('from pytest import approx')
with_pba = pff.voluptuous.Namespace(
        'import numpy as np',
        'from numpy import log',
        'from pba_waeber2013 import *',
)

def test_function_wrapper():
    f = FunctionWrapper(
            lambda x, a, b: x + a + b,
            (1,), {'b': 2},
            2,
    )

    assert f(1) == approx(4)
    assert f.n == 1
    assert f.x_obs == approx([1])
    assert f.f_obs == approx([4])

    assert f(2) == approx(5)
    assert f.n == 2
    assert f.x_obs == approx([1, 2])
    assert f.f_obs == approx([4, 5])

    with raises(IndexError):
        f(3)

def test_posterior_attrs():
    posterior = PosteriorDist([0, 1, 0], [0, 0], 0)

    # These attributes should exclude allocated-but-unused elements:
    assert posterior.xs == approx([0, 1])
    assert posterior.log_ps == approx([0])

@pff.parametrize(
        schema=Schema({
            'posterior': with_pba.eval(defer=True),
            'x': Coerce(float),
            'x_i': Coerce(int),
        }),
)
def test_posterior_median(posterior, x, x_i):
    posterior = posterior.eval()
    x_actual, x_i_actual = posterior.median()

    assert x_actual == approx(x)
    assert x_i_actual == x_i

@pff.parametrize(
        schema=Schema({
            'posterior': with_pba.eval(defer=True),
            'x': Coerce(float),
            'x_i': Coerce(int),
            'sign_guess': Coerce(int),
            'p_c': Coerce(float),
            'slope': Coerce(int),
            'expected': {
                'xs': with_py.eval,
                'ps': with_py.eval,
                'n': Coerce(int),
            },
        }),
)
def test_posterior_update(posterior, x, x_i, sign_guess, p_c, slope, expected):
    posterior = posterior.eval()
    posterior.update(x, x_i, sign_guess, p_c, slope)

    assert posterior.n == expected['n']
    assert posterior.xs == approx(expected['xs'])
    assert posterior.log_ps == approx(np.log(expected['ps']))

@pff.parametrize(
        schema=Schema({
            'ci': with_pba.eval(defer=True),
            Optional('n', default=-1): Coerce(int),
            'width': Coerce(float),
            'width_seq': Coerce(float),
        }),
)
def test_ci_width(ci, n, width, width_seq):
    ci = ci.eval()
    assert ci.width(n) == approx(width)
    assert ci.width_seq(n) == approx(width_seq)

@pff.parametrize(
        schema=Schema({
            'ci': with_pba.eval(defer=True),
            'posterior': with_pba.eval(defer=True),
            'p_c': Coerce(float),
            'alpha': Coerce(float),
            'expected': with_py.eval,
            'expected_seq': with_py.eval,
        }),
)
def test_ci_update(ci, posterior, p_c, alpha, expected, expected_seq):
    ci = ci.eval()
    posterior = posterior.eval()

    ci.update(posterior, p_c, alpha)
    assert ci.ci == approx(np.array(expected))
    assert ci.ci_seq == approx(np.array(expected_seq))

@pff.parametrize(
        schema=Schema({
            'random_walk': with_py.eval,
            'p_c': Coerce(float),
            'n_max': Coerce(int),
            'expected': Coerce(int),
        }),
)
@pytest.mark.parametrize('sign', [1, -1])
def test_determine_sign(random_walk, p_c, n_max, sign, expected):
    it = iter(random_walk)

    def f(x):
        return sign * next(it)

    assert determine_sign(f, 0, p_c, n_max) == sign * expected

@pff.parametrize(
        schema=Schema({
            'x': Coerce(float),
            'atol': with_py.eval,
            'rtol': with_py.eval,
            'expected': Coerce(float),
        }),
)
def test_get_tol(x, atol, rtol, expected):
    assert get_tol(x, atol, rtol) == approx(expected)

@pff.parametrize(
        schema=Schema({
            'f': with_py.exec(get='f'),
            'noise': Coerce(float),
            'a': with_py.eval,
            'b': with_py.eval,
            'kwargs': {str: with_py.eval},
            **with_pba.error_or({
                Optional('expected', default='None'): with_approx.eval,
                Optional('converged', default='None'): with_py.eval,
                Optional('n_obs', default=0): Coerce(int),
                Optional('n_post', default=0): Coerce(int),
            }),
        }),
)
@pytest.mark.parametrize('use_noise', [False, True])
def test_pba_waeber2013_params(
        f, a, b, noise, use_noise, kwargs,
        expected, converged, n_obs, n_post, error
):
    rng = np.random.default_rng(0)
    def g(x, *args, **kwargs):
        return f(x, *args, **kwargs) + use_noise * rng.normal(0, noise)

    with error:
        result = pba_waeber2013(g, a, b, **kwargs)

        # import matplotlib.pyplot as plt

        # fig, axes = plt.subplots(1, 3)
        # axes[0].plot(result.x_obs, result.f_obs, '+')

        # i = np.arange(result.ci.shape[1])
        # axes[1].plot(i, result.ci[0])
        # axes[1].plot(i, result.ci[1])

        # x = np.diff(result.x_post) / 2 + result.x_post[:-1]
        # axes[2].plot(x, result.p_post)

        # plt.tight_layout()
        # plt.show()

        if converged is not None:
            assert result.converged == converged
        if expected is not None:
            assert result.x == expected
        if n_obs:
            assert len(result.x_obs) == n_obs
            assert len(result.f_obs) == n_obs
        if n_post:
            assert len(result.x_post) == n_post + 2
            assert len(result.p_post) == n_post + 1
