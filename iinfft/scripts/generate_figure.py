import numpy as np
import matplotlib.pyplot as plt
from iinfft.iinfft import *
plt.style.use('tableau-colorblind10')
import pandas as pd
#import sym_matrix
import finufft

import matplotlib
matplotlib.use('TkAgg')

df = pd.read_csv('iinfft/data/T.Suelo.csv')
Ln = df.shape[0]
data_raw = df.iloc[0:,1:].to_numpy() #keep the missing values

N = 1024
t = 2*np.pi*np.linspace(-0.5,0.5,Ln,endpoint=False).astype(np.float32)
#inverse_mat = np.zeros((N,df.shape[1]-1),dtype="complex128")
w = w_sobolev(N,1,2,1e-4).astype(np.float32)

# N = 64
# k = kgrid(N)

# kernels = {
#     "Dirichlet": w_dirichlet(N),
#     "Fejér": w_fejer(N),
#     "Jackson (p=2)": w_jackson(N, p=2),
#     "Gaussian (σ=0.25)": w_gaussian(N, sigma=0.25),
#     "Raised cosine (α=1)": w_raised_cosine(N, alpha=1.0),
#     "B-spline-like (β=2)": w_bspline_like(N, beta=2),
#     "Sobolev (a=1,b=2,γ=1e-2)": w_sobolev(N, a=1.0, gamma=1e-2)
# }

# for name, w in kernels.items():
#     plt.figure()
#     plt.plot(k, N*np.abs(np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(w)))))
#     plt.title(f"Real part of {name} kernel (N={N})")
#     plt.xlabel("k (centered frequency index)")
#     plt.ylabel("Re{w_k}")
#     plt.grid(True)
#     plt.show()

dat = data_raw[:,4].astype(np.complex64)

idx = dat != -9999
# if sum(idx) % 2 != 0:
#     idx = change_last_true_to_false(idx)

dat_clean = dat[idx].copy().astype(np.complex64)
h_k = -(N // 2 ) + np.arange(N)
AhA = compute_sym_matrix_optimized(t[idx],h_k)

ftot, _, _, _ = infft(t[idx], dat[idx] - np.mean(dat[idx]),N=N,AhA=AhA,w=w,gpu=False,return_adjoint=False)
ytot = finufft.nufft1d2(t,ftot,isign=+1) + np.mean(dat[idx])

ax = plt.gca()

#Scatter plot of observed data
ax.scatter(t[idx],dat[idx],s=0.1,c='k')

#Scatter plot of iNFFT
infft_line, = ax.plot(t,np.abs(ytot).astype(np.float64),c='C4',label='iNFFT')

#Scatter plot of truncated iFFT
fapprox, _, _, _ = infft(t[idx], dat[idx] - np.mean(dat[idx]),N=N,AhA=AhA,w=w,approx=True,gpu=False,return_adjoint=False)
yapp = finufft.nufft1d2(t,fapprox,isign=+1) + np.mean(dat[idx])

ifft_line, = ax.plot(t,np.abs(yapp).astype(np.float64),c='C1',label='iFFT')
ax.legend(handles=[infft_line,ifft_line])

plt.xlabel(r"Normalized time values $t \in [-\pi,\pi)$")
plt.ylabel(r"Temperature ($^\circ$C)")
plt.title(r"Interpolative iNFFT on irregularly sampled remote sensor data")
plt.savefig(r'result_figure.png')

print('MAPE: iNFFT')
print(len(idx)**(-1) * np.sum(np.abs(np.divide(np.abs(ytot[idx]) - dat[idx],dat[idx]))))

print('MAPE: iFFT')
print(len(idx)**(-1) * np.sum(np.abs(np.divide(np.abs(yapp[idx]) - dat[idx],dat[idx]))))
