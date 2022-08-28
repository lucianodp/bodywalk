# Introduction
BodyWalk is a python library implementing popular random walk techniques for sampling
uniformly over convex bodies. More precisely, this package offers the following functionalities:

1) Efficient implementation of several random walk algorithms, including Ball Walk, Hit-and-Run, and Billiard Walk;
2) Support for general convex bodies, including rectangles, balls, and polytopes;
3) Implementation of rounding techniques for improved mixing time.

# Example
```
from bodywalk.sampling import ball_walk, billiard_walk, hit_and_run
from bodywalk.geometry import Polytope

convex_body = Polytope([[1, 0], [-1, 0], [0, 1], [0, -1]], [0.5]*4)  # Square of side 1 centered at (0, 0)
initial_sample = [0, 0]  # Initial point to start the Markov Chain
random_state = 42  # RNG seed

chain = ball_walk(convex_body, initial_sample, delta=0.5, random_state=random_state)
# chain = billiard_walk(convex_body, initial_sample, tau=0.5, random_state=random_state)
# chain = hit_and_run(convex_body, initial_sample, random_state=random_state)

for sample in chain:
  # process the new generated sample
```

# Dependencies
This module was tested using the following dependencies:
- Python >= 3.9
- Numpy >= 1.21.4
- Pytest >= 6.2.5

# License
This project is released under the BSD-3 Clause license.

# References
[1] L. Lovász. **An Algorithmic Theory of Numbers, Graphs and Convexity**, volume 50 of *CBMS-NSF Regional Conference Series in Applied Mathematics*. Society for Industrial and Applied Mathematics, 1986. ISBN 9781611970203.

[2] L. Lovász. **Hit-and-run mixes fast**. *Mathematical Programming*, 86(3):443–461, 1999. ISSN 0025-5610. doi: 10.1007/s101070050099.

[3] B.T. Polyak and E.N. Gryazina. **Billiard walk-a new sampling algorithm for control and optimization**. *IFAC Proceedings Volumes*, 47(3):6123-6128, 2014.
