# SBOT: A Sample Based Optimal Transport solver
This Python package contains code to solve sample based optimal transport problems.
An introduction to sample based optimal transport can be found here https://math.nyu.edu/faculty/tabak/publications/Kuang-Tabak.pdf .

Requirements are numpy, scipy. It is recommended to have a Python version >= 3.3

For now the package includes:
* igd_sover : A nonlinear minimization solver based on implicit gradient descent. Convergence to a global minimizer isn't guaranteed.
* local_sot : A solver for sample based __local__ optimal transport.
* sot : A solver for sample based optimal transport.
