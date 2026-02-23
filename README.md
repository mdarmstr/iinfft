# Inverse Interpolative Non-Uniform Fast Fourier Transform (i-iNFFT)

Interpolative inverse non-uniform fast fourier transform. Interpolates missing data (labelled as NaNs) based on a smoothed series of Fourier coefficients according to a user-defined kernel. The Fejer and Sobolev kernels are included in this module.

# Theory

The type-I Discrete Non-Uniform Transform (NDFT) is defined as the following operation:
$$h_k = \sum_{M/2-1}^{M/2} f(x_j) e^{-2\pi\mathbf{i}k\frac{x_j}{M}}$$

Where $h_k$ refers to a uniform set of Fourier coefficients, for a specified number $N$ of trigonometric polynomials. $f(x_j)$ refers to obsevational data in the time domain taken at irregularly sampled points: $x_j$ which are measured in the domain $[-\frac{M}{2}, \frac{M}{2})$

Which can be performed in $\mathcal{O}(N log N + |log (1/\epsilon) | M)$ complexity using the Non-Uniform Fast Fourier Transform described by Potts et al (Potts, Daniel, Gabriele Steidl, and Manfred Tasche. "Fast Fourier transforms for nonequispaced data: A tutorial." Modern Sampling Theory: Mathematics and Applications (2001): 247-270.). Note that for consistency with common convention, we define the forward transform as above, and the adjoint via a sign change in the exponential term. This is opposite to the convention utilised by Potts in the mathematical body of literature.

Unlike the Fast Fourier Transform and its inverse in the equidistant case, the normalisation for an inverse transform is not known explicitly and relies on the implicit inverse of the self-adjoint product of the forward and adjoint transformation matices: $AA^H$. In order to calculate a series of Fourier coefficients that is invertible via its adjoint product, the coefficients must account for the inverse of $AA^H$. This is done using the LU decomposition of the inverse product, via a weighted filter to avoid fitting to a discontinuous function. 

Formally, this package performs the inverse adjoint non-uniform fast fourier transform as defined by our convention via the minimisation of the cost function:

$${argmin}_{\hat{h}_k}||f(x_j) - A^H\hat{h}_k||_2^2 + ||\hat{h}_k||^2_{\hat{W}^{-1}} $$

# Installation (CPU)

```
git clone https://github.com/mdarmstr/iinfft
cd iinfft
pip install .
```
# Installation (CUDA)

```
git clone https://github.com/mdarmstr/iinfft
cd iinfft
pip install .[gpu]
```

# Basic usage
This module contains functionality for 1D and 2D Numpy arrays. An example of an interpolative transform for 1D is shown below. The number of Fourier coefficients, N must be selected. Since the algorithm executes in MlogN + N^2 time complexity, a large number of Fourier coefficients will significantly impact the execution time. The Sobolev kernel is selected for the smoothing. The AhA is the inner product matrix that is calculated using the Numpy C API. `ftot` are the calculated, dampened frequency coefficeints and `ytot` are the interpolated values over the entire array following the adjoint (i.e. un-normalized transform).

```
import numpy as np
import finufft
from iinfft.iinfft import *

data_raw = np.load("data/T.Suelo.csv")

N = 1024
t = 2*np.pi*np.linspace(-0.5,0.5,Ln,endpoint=False)
w = w_sobolev(N,1,2,1e-2)

dat = data_raw[:,0]

idx = dat != -9999

dat_clean = dat[idx].copy()
h_k = -(N // 2 ) + np.arange(N)
AhA = compute_sym_matrix_optimized(t[idx],h_k)

ftot, _, _, _ = infft(t[idx], dat[idx] - np.mean(dat[idx]),N=N,AhA=AhA,w=w)
ytot = finufft.nufft1d2(t,ftot,isign=+1) + np.mean(dat[idx],isign=+1)
```
A similar procedure can be followed for 2D arrays, which calculate the parallel iiNFFTSs followed by parallel FFTs on the opposite mode.

```
N = 64
w = w_sobolev(N,1,2,1e-2)
transformed_data, mtot = infft_2d(image, N,w=w)

# Adjoint transform (reconstruction)
reconstructed_data = adjoint_transform_2d(
    transformed_data,
    mtot,
    data_shape=image.shape)
```
# Authors
Computational Data Science (CoDaS) Lab, University of Granada
Michael Sorochan Armstrong (mdarmstr@go.ugr.es)
José Camacho

## Copyright

Copyright (C) 2026  Universidad de Granada
 
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

