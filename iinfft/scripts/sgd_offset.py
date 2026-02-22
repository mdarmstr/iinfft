import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from iinfft.iinfft import *
import finufft
#from batch_job import test_infft
import pandas as pd

plt.style.use('tableau-colorblind10')

def test_infft(dat, idx, Ln, N, w):
    stp = np.random.randint(0, Ln)
    t = np.linspace(-0.5, 0.5, Ln, endpoint=False)
    Mn = int(Ln / (t[stp] + 0.5) / (1 - 1/Ln))
    #Mn = int(Ln / (t[stp] + 2*np.pi*0.5) / (2*np.pi - 2*np.pi / Ln))
    #Mn = max(Ln, int(Ln / ((t[stp]/(2*np.pi)) + 0.5) / (1 - 1/Ln)))
    
    i_obs = np.flatnonzero(idx)
    x_msr = 2*np.pi*(i_obs / Mn - 0.5)

    h_k = -(N // 2 ) + np.arange(N)
    AhA1 = compute_sym_matrix_optimized(x_msr, h_k)
    _, _, _, err = infft(x_msr, dat[i_obs] - np.mean(dat[i_obs]), N=N, AhA=AhA1, w=w, return_adjoint=True)
    return (err, stp)

if __name__ == '__main__':

    df = pd.read_csv('iinfft/data/T.Suelo.csv')
    Ln = df.shape[0]
    smplR = 1800
    data_raw = df.iloc[0:,1:].to_numpy() #keep the missing values
    inverse_mat = np.zeros_like(data_raw,dtype="complex128")
    residue_mat = np.zeros_like(data_raw,dtype="float64")
    rec_mat = np.zeros_like(data_raw,dtype="float64")
    mni = np.zeros((df.shape[1]-1,1))

    N = 32
    t = 2*np.pi*np.linspace(-0.5,0.5,Ln,endpoint=False)
    inverse_mat = np.zeros((N,df.shape[1]-1),dtype="complex128")
    #w = fjr(N)
    w = w_sobolev(N,1,2,1e-1)

    dat = data_raw[:,0]

    idx = dat != -9999
    if sum(idx) % 2 != 0:
        idx = change_last_true_to_false(idx)

    h_k = -(N // 2 ) + np.arange(N)
    AhA1 = compute_sym_matrix_optimized(t[idx],h_k)
    ftot, _, _, _ = infft(t[idx], dat[idx] - np.mean(dat[idx]),N=N,AhA=AhA1,w=w)
    ytot_mean = np.abs(finufft.nufft1d2(t,ftot,isign=+1) + np.mean(dat[idx]))

    num_cores = 10 #multiprocessing.cpu_count() - 1

    q = np.zeros(5)
    err_best = np.inf
    stp_best = 0
    stp_iter = Ln
    eta = 0.25
    itrs = []
    errs = []
    epochs = 12
    results = [[None, None] for _ in range(num_cores)]

    for ii in range(epochs):
        for jj in range(num_cores):
            results[jj] = test_infft(dat,idx,Ln,N,w)
            args = [(dat,idx,Ln,N,w) for _ in range(num_cores)]
            #results = pool.starmap(test_infft,args)

        for jj in range(num_cores):
            if results[jj][0] < err_best:
                err_best = results[jj][0]
                stp_best = results[jj][1]
                stp_iter = int(stp_iter - eta * (stp_iter - stp_best))
        
        itrs.append(stp_iter)
        errs.append(err_best)

        print(f"epoch {ii} complete, error {errs[-1]}, stp_iter {stp_iter}")

    print("")
    #Plotting the results
    #t = np.linspace(-0.5,0.5,Ln,endpoint=False)
    tnorm = np.linspace(-0.5, 0.5, Ln, endpoint=False)
    Mn = int(Ln / (tnorm[stp_iter] + 0.5) / (1 - 1/Ln))

    i_obs = np.flatnonzero(idx)
    x_meas = 2*np.pi*(i_obs / Mn - 0.5)
    y_meas = dat[i_obs]
    y0 = np.mean(y_meas)

    AhA1 = compute_sym_matrix_optimized(x_meas, h_k)
    fshift, _, _, _ = infft(x_meas, y_meas - y0, N=N, AhA=AhA1, w=w)

    x_full = 2*np.pi*(np.arange(Mn) / Mn - 0.5)
    ytot_shift = np.real(finufft.nufft1d2(x_full, fshift, isign=+1)) + y0

    # Creating subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Plotting scatter and line on the unshifted data
    axs[0].scatter(t[idx], dat[idx], s=0.1,c='k',label='measurements')
    axs[0].plot(t, ytot_mean, color='C1', label='unshifted')
    axs[0].set_title(r'Naive time labels')
    axs[0].set_xlabel(r"Normalized time values $t \in [-\pi,\pi)$")
    axs[0].set_ylabel(r"Temperature ($^\circ$C)")
    axs[0].legend()

    # Plotting scatter and line on the second subplot
    axs[1].scatter(x_meas, y_meas, s=0.1, c='k', label='measurements')
    axs[1].plot(x_full, ytot_shift, color='C2', label='shifted')
    axs[1].set_title(r'SGD shifted time labels')
    axs[1].set_xlabel(r"Normalized time values $t \in [-\pi,\pi)$")
    axs[1].set_ylabel(r"Temperature ($^\circ$C)")
    axs[1].legend()

    # Adjusting layout and displaying the plots
    plt.tight_layout()
    plt.show()
    plt.savefig("sgd_comparison.png")



