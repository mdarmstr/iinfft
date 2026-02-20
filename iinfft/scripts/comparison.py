import numpy as np
import matplotlib.pyplot as plt
from iinfft.iinfft import *
plt.style.use('tableau-colorblind10')
import pandas as pd
#import sym_matrix
import time
import matplotlib
matplotlib.use('TkAgg')

## For the purposes of comparing the explicit calculation of the inner product (Toeplitz) with the O(N^2) routine.

df = pd.read_csv('iinfft/data/T.Suelo.csv')
Ln = df.shape[0]
smplR = 1800
data_raw = df.iloc[0:,1:].to_numpy() #keep the missing values
inverse_mat = np.zeros_like(data_raw,dtype="complex128")
residue_mat = np.zeros_like(data_raw,dtype="float64")
rec_mat = np.zeros_like(data_raw,dtype="float64")
mni = np.zeros((df.shape[1]-1,1))

N = 1024
t = np.linspace(-0.5,0.5,Ln,endpoint=False)
inverse_mat = np.zeros((N,df.shape[1]-1),dtype="complex128")
#w = fjr(N)
w = w_sobolev(N,1,2,1e-2)

dat = data_raw[:,0]

idx = dat != -9999
if sum(idx) % 2 != 0:
    idx = change_last_true_to_false(idx)

h_k = -(N // 2 ) + np.arange(N)
AhA = compute_sym_matrix_optimized(t[idx],h_k,gpu=False)
AhA1 = compute_sym_matrix_optimized(t[idx],h_k,gpu=True).cpu().numpy()
# A = ndft_mat(t[idx],N)
# AhA1 = A.H@A

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# First plot: Imaginary part of sym_mat
axs[0].imshow(np.imag(AhA), cmap='viridis', aspect='auto')
axs[0].set_title('Imaginary Part of sym_mat')
axs[0].set_xlabel('Columns')
axs[0].set_ylabel('Rows')

# Second plot: Imaginary part of A.H@A
axs[1].imshow(np.imag(AhA1), cmap='viridis', aspect='auto')
axs[1].set_title('Imaginary Part of A.H@A')
axs[1].set_xlabel('Columns')
plt.tight_layout()
plt.savefig('side_by_side_mult.png')

AhA1_np = np.asarray(AhA1)

diff = AhA - AhA1_np
print("max |diff|:", np.max(np.abs(diff)))
print("max |imag diff|:", np.max(np.abs(np.imag(diff))))
print("max |real diff|:", np.max(np.abs(np.real(diff))))
print("relative Fro error:", np.linalg.norm(diff) / np.linalg.norm(AhA1_np))