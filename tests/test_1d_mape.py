import numpy as np
import pandas as pd
import finufft

from iinfft.iinfft import (
    w_sobolev,
    infft,
    compute_sym_matrix_optimized,
    change_last_true_to_false,
)

def test_1d_infft_beats_ifft_mape():
    df = pd.read_csv("iinfft/data/T.Suelo.csv")
    data_raw = df.iloc[:, 1:].to_numpy()
    Ln = data_raw.shape[0]

    # Smaller N for CI speed
    N = 256
    t = 2 * np.pi * np.linspace(-0.5, 0.5, Ln, endpoint=False).astype(np.float32)

    w = w_sobolev(N,1,2,1e-4).astype(np.float32)

    # Same column choice as your script (col index 4 of data_raw)
    dat = data_raw[:, 4].astype(np.complex64)

    idx = dat != -9999
    # if idx.sum() % 2 != 0:
    #     idx = change_last_true_to_false(idx)

    h_k = -(N // 2) + np.arange(N)
    AhA = compute_sym_matrix_optimized(t[idx], h_k, gpu=False)

    mean = np.mean(dat[idx])

    fk_infft, _, _, _ = infft(t[idx], dat[idx] - mean, N=N, AhA=AhA, w=w, gpu=False)
    y_infft = finufft.nufft1d2(t, fk_infft, isign=+1) + mean

    fk_ifft, _, _, _ = infft(t[idx], dat[idx] - mean, N=N, AhA=AhA, w=w, approx=True, gpu=False)
    y_ifft = finufft.nufft1d2(t, fk_ifft, isign=+1) + mean

    # MAPE on observed points only
    mape_infft = np.mean(np.abs((y_infft[idx] - dat[idx]) / dat[idx]))    
    mape_ifft  = np.mean(np.abs((y_ifft[idx]  - dat[idx]) / dat[idx]))

    # Loose inequality: should consistently hold given your figure script behavior
    assert mape_infft < mape_ifft