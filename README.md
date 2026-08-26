# ReLU over Poulpy

> **A FHERMA reference implementation on [Poulpy](https://github.com/poulpy-fhe/poulpy) (Apache-2.0).**
> Implements [`relu` / `f64@1.0.0`](https://fherma.io/kernels/relu).

max(0, x) interpolated directly — ReLU is continuous, so no ramp trick is needed. One degree-1023 Chebyshev in a single BSGS pass keeps the worst error near the kink around 4·10⁻⁴, inside the specification's 1e-3 tolerance across the whole domain: share 100% against a bar of 85%. Ten rescales of a 330-bit budget on a 360-bit torus at ring degree 16384; no rotation keys.

Run: see solution/README.md.

License: Apache-2.0.
