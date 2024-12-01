# BOCD

## Overview

This package implements and extends the Bayesian Online Changepoint Detection (BOCD) algorithm, which is described in a paper by Adams and MacKay ([1]).

BOCD is a Bayesian method for detecting **changepoints** in time series data. It does this by modeling the posterior
distribution $p\left(r_{t}\mid x_{1:t}\right)$ of the so called **run length** $r_{t}$ at each point in time $t = 1, ...$, given the
history of observations $(x_{1},...,x_{t})=:x_{1:t}$.

The main features of this implementation are:

- Extension of the algorithm to compute "look-ahead" posterior probabilities $p\left(r_{t}\mid x_{1:t+h}\right)$ for any $h\geq0$,
  whereas the original algorithm computes this posterior for $h=0$.
  This equips BOCD with offline changepoint
  detection capabilities (see [derivations](docs/derivations.pdf)).
- Support for arbitrary prior distributions $p\left(r_{0}\right)$ for the run length at time $t=0$.
- Support for arbitrary changepoint probability distributions $p\left(r_{t}=0\mid r_{t-1}\right)$.
- Any probability distribution $p_{\mathrm{gap}}(g)$ for the gap $g$ between consecutive changepoints
  can easily be transformed into a changepoint probability distribution $p\left(r_{t}=0\mid r_{t-1}\right)$.
- Support for arbitrary predictive distributions (times series models) $p\left(x_{t}\mid r_{t-1},x_{t-r_{t}:t-1}\right)$.
- Changepoint as well as predictive distributions can depend on time.
- Thresholding of run length probabilities in the tail of the distribution to save computational time and memory.
  This idea is mentioned in section 2.4 of the paper.

## Examples

Examples can be found in a [Jupyter notebook](examples/Basics.ipynb).

## Installation

```bash
git clone https://github.com/BayesOnBikes/bocd.git
poetry install
```

## Requirements

The code was written in Python 3.13.

For BOCD:

- numpy
- scipy

For the examples, the following packages are additionally required:

- matplotlib
- notebook (Jupyter)

## Roadmap

- **Additional Time Series Models**: Implement more models.
- **Unit Tests**: Develop a comprehensive test suite to ensure code reliability.

## Acknowledgments

- **Gregory Gundersen**: For his informative blog post and code examples ([2]).

## References

- [1]: The original paper can be found here: https://arxiv.org/pdf/0710.3742v1.
- [2]: This blog post and the code associated with it is an excellent resource: https://gregorygundersen.com/blog/2019/08/13/bocd/.