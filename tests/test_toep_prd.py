import numpy as np

from iinfft.iinfft import compute_sym_matrix_optimized, ndft_mat

def test_toeplitz_matches_explicit_AhA():
    rng = np.random.default_rng(0)

    N = 128
    M = 300

    # irregular samples in [-pi, pi)
    #t = rng.uniform(-np.pi, np.pi, size=M).astype(np.float64)
    t = 2*np.pi * rng.uniform(-0.5, 0.5, size=M)
    h_k = -(N // 2) + np.arange(N)

    AhA_toeplitz = compute_sym_matrix_optimized(t, h_k, gpu=False)

    A = ndft_mat(t, N)
    AhA_explicit = np.asarray(A.conj().T @ A)

    diff = AhA_toeplitz - AhA_explicit
    rel = np.linalg.norm(diff) / np.linalg.norm(AhA_explicit)

    assert rel < 1e-10