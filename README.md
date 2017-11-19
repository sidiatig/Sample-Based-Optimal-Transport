# SamBa OT: A Sample Based Optimal Transport solver

## Introduction
This Python package contains code to solve sample based optimal transport problems.
An introduction to sample based optimal transport can be found here https://math.nyu.edu/faculty/tabak/publications/Kuang-Tabak.pdf .

The code is still in an alpha version, and optimization/new packages will be added in the future.

Requirements are the numpy, scipy packages. It is recommended to have a Python version >= 3.3

## Contents
For now the package includes:
* igd_sover : A nonlinear minimization solver based on implicit gradient descent. Convergence to a global minimizer isn't guaranteed.
* local_sot : A solver for sample based __local__ optimal transport.
* sot : A solver for sample based optimal transport.
* ot_map : Reconstructs the global optimal transport map from the outputs of the sot solver.

## Installation
Download the file sbot-0.1.tar.gz located in the 'dist' folder. Run the command 'pip install sbot-0.1.tar.gz' after changing your working directory to where the file is downloaded.

## Examples
Examples and explanations are included in the SampleBasedOT.ipynb Jupyter notebook in the 'notebooks' folder.
