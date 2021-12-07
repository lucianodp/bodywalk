# Introduction
BodyWalk is a python library implementing popular random walk techniques for sampling
uniformly over convex bodies. More precisely, this package offers the following functionalities:

1) Efficient implementation of several random walk algorithms, including Ball Walk, Hit-and-Run, Dikin Walk, and others
2) Support for general convex bodies, including rectangles, balls, and polytopes.
3) Implementation of rounding techniques for improved mixing time

# Dependencies
This module has the following dependencies:
- Python >= 3.9.9
- Numpy >= 1.21.4
- Pytest >= 6.2.5

# License
This project is released under the MIT license.

# References
[1] L. Lovász. **An Algorithmic Theory of Numbers, Graphs and Convexity**, volume 50 of *CBMS-NSF Regional Conference Series in Applied Mathematics*. Society for Industrial and Applied Mathematics, 1986. ISBN 9781611970203.

[2] L. Lovász. **Hit-and-run mixes fast**. *Mathematical Programming*, 86(3):443–461, 1999. ISSN 0025-5610. doi: 10.1007/s101070050099.
