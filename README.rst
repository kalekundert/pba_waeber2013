****************
PBA [Waeber2013]
****************

.. image:: https://img.shields.io/pypi/v/pba_waeber2013.svg
   :alt: Last release
   :target: https://pypi.python.org/pypi/pba_waeber2013

.. image:: https://img.shields.io/pypi/pyversions/pba_waeber2013.svg
   :alt: Python version
   :target: https://pypi.python.org/pypi/pba_waeber2013

.. image:: 
   https://img.shields.io/github/workflow/status/kalekundert/pba_waeber2013/Test%20and%20release/master
   :alt: Test status
   :target: https://github.com/kalekundert/pba_waeber2013/actions

.. image:: https://img.shields.io/coveralls/kalekundert/pba_waeber2013.svg
   :alt: Test coverage
   :target: https://coveralls.io/github/kalekundert/pba_waeber2013?branch=master

.. image:: https://img.shields.io/github/last-commit/kalekundert/pba_waeber2013?logo=github
   :alt: Last commit
   :target: https://github.com/kalekundert/pba_waeber2013

This is an implementation of the probabilistic bisection algorithm (PBA) 
described by Rolf Waeber in his 2013 PhD thesis.  PBA is a 1D stochastic 
root-finding algorithm.  This means that it is meant to find the point where a 
noisy function (i.e. a function that may return different values each time its 
evaluated at the same point) crosses the x-axis.  More precisely, given *g(x) = 
f(x) + ε(x)*, the goal is to find *x* such that *E[g(x)] = 0*, where 
*f(x)* is the function we are interested in, *ε* is a normally 
distributed noise function with median 0, *g(x)* is the only way we can 
observe *f(x)*, and *E[g(x)]* is the expected value of 
*g(x)*. 

This algorithm works by repeatedly evaluating the noisy function at a single 
*x* until the probability that we know the true sign of *f(x)* 
exceeds a specified threshold.  This information is used to build a Bayesian 
posterior distribution describing the location of the root.  The next point to 
evaluate is then chosen from this distribution.  There are two features that 
make this algorithm unique:

1. It provides a true confidence interval in addition to the root itself.

2. It places very little restraints on the form of the noise function.

However, there are some caveats to be aware of:

1. In my experience, the algorithm rarely converges.  This is due to the fact 
   that the confidence intervals only narrow as new x-coordinates are sampled, 
   but the closer the algorithm gets to the root, the more time it spends 
   sampling each x-coordinate.  That said, the algorithm seems to find the root 
   very accurately even when it doesn't converge.

2. The algorithm isn't very numerically stable.  The posterior distribution is 
   represented using two arrays: one for the bin edges and one for the bin 
   heights.  As the algorithm progresses, the bins near the root get very thin 
   and very tall, which leads to loss-of-precision and overflow issues.  My 
   only advice on how to avoid these problems is to limit the number of 
   iterations.  It might also help to increase the *p_c* parameter.

Note that I myself only have a rudimentary understanding of the math behind 
this algorithm.  This implementation is based on scripts I received from the 
authors, and I tried to test my code as well as possible to convince myself 
that it's doing the right thing, but I'm outside my comfort zone here.

Installation
============
Install from PyPI::

  $ pip install pba_waeber2013

Usage
=====
This module provides a single public function::

  >>> from pba_waeber2013 import pba_waeber2013
  >>> pba_waeber2013(
  ...         f, a, b, *
  ...         tol=None,
  ...         rtol=None,
  ...         alpha=0.05,
  ...         p_c=0.65,
  ...         maxiter=1000,
  ...         check_bounds=False,
  ...         slope=None,
  ...         args=None,
  ...         kwargs=None,
  ... )

Below are brief descriptions of the parameters:

*f*
  The stochastic function of interest.  This function can take any number of 
  arguments (see *args* and *kwargs*), but the first should be the *x* 
  coordinate to evaluate the function at.

*a b*
  The lower and upper bounds, respectively, on where the root can occur.  There 
  must be exactly one root in this interval.

*tol*
  How narrow the confidence interval needs to be in order for the algorithm to 
  be converged.  Note that if neither *tol* nor *rtol* are specified, the 
  algorithm will just run for the maximum number of iterations.

*rtol*
  Similar to *tol*, but multiplied by the estimated root.

*alpha*
  A parameter controlling the width of the confidence intervals.  Specifically, 
  the confidence level is given by *1 − alpha*.  In other words, for 95% 
  confidence intervals, set *alpha* to 0.05.

*p_c*
  Repeatedly evaluate the function at each *x* coordinate until we can reject 
  the null hypothesis that *f(x) = 0* with probability *p_c*.  At that point, 
  the sign of *f(x)* will be taken as whichever sign we observed most often.

*maxiter*
  The maximum number of function calls to make.

*check_bounds*
  Evaluate the function at the bounds *a* and *b* before starting the 
  bisection.  This achieves two things.  First, it checks that the bounds have 
  different signs.  If this is not the case, then there are either 0 or >1 
  roots in the interval, and so this algorithm is not applicable.  Second, it 
  determines the slope of the function.  If the bounds are not checked, the 
  *slope* parameter must be manually specified.

*slope*
  +1 if the function is increasing (i.e. negative before the root and positive 
  after) or -1 if the function is decreasing.  A slope must be given unless 
  *check_bounds* is True (in which case it will be calculated internally).

*args*
  Any extra arguments to pass to the function on each evaluation.

*kwargs*
  Any extra keyword arguments to pass to the function on each evaluation.
  
The return value is an object with the following attributes:

*x*
  The location of the root.

*x_obs*
  All of the *x* coordinates where the function was evaluated.

*f_obs*
  The result of evaluating the function at each of the above *x* coordinates.

*x_post*
  The bin edges of the posterior distribution.

*log_p_post*
  The natural logarithms of the bin heights of the posterior distribution.  
  Logarithms are used to avoid multiplication overflows.

*ci*
  The confidence interval evaluated independently after a sign is determined 
  for each coordinate.  Note that these intervals can grow and shrink over 
  time.  See [Waeber2013] §3.3 for more information.

*ci_seq*
  The sequential confidence interval.  Unlike the *ci* intervals, these are 
  guaranteed to never expand.  For that reason, these are the intervals used to 
  check for convergence.

*converged*
  True if the algorithm terminated because the confidence interval grew 
  narrower than the given tolerance, False if the algorithm terminated because 
  it reached the maximum number of iterations.

References
==========
- Waeber R. (2013) "Probabilistic bisection search for stochastic 
  root-finding."

- Frazier PI, Henderson SG, Waeber R (2016) "Probabilistic bisection converges 
  almost as quickly as stochastic approximation", arXiv:1612.03964

- Robbins H and Siegmund D. (1974) "The expected sample size of some tests of 
  power one", The Annals of Statistics, 2(3), pp. 415–436.  
  doi:10.1214/aos/1176342704.
  
  

  

