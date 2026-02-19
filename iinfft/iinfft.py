import numpy as np
import finufft
from scipy.linalg import lu,toeplitz

def is_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        import cufinufft
        _ = torch.cuda.current_device()
        return True
    except Exception:
        return False

def _torch_device(gpu: bool):
    import torch
    return torch.device("cuda") if (gpu and torch.cuda.is_available()) else torch.device("cpu")

def _to_torch(x, device, dtype=None):
    import torch
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype) if dtype is not None else x.to(device=device)
    # numpy -> torch
    return torch.as_tensor(x, device=device, dtype=dtype) if dtype is not None else torch.as_tensor(x, device=device)

def _pick_nufft_backend(gpu: bool):
    """
    Returns (use_gpu_backend, nufft_module)
    """
    if not gpu:
        return False, finufft

    try:
        import torch
        if not torch.cuda.is_available():
            return False, finufft
        import cufinufft  # must be installed
        _ = torch.cuda.current_device()  # forces CUDA init
        return True, cufinufft
    except Exception:
        return False, finufft

def ndft_mat(x,N):
    #non-equispaced discrete Fourier transform Matrix
    k = -(N // 2) + np.arange(N)
   
    return np.asmatrix(np.exp(1j * np.outer(k,x[:,np.newaxis])).T)

def change_last_true_to_false(arr):
    
    arr = np.asarray(arr)
    indices = np.where(arr)[0]
    if len(indices) > 0:
        last_true_index = indices[-1]
        arr[last_true_index] = False
    
    return arr

# ============================================================
# Frequency-domain kernels/windows (CENTERED ORDER)
# k = -(N//2), ..., (N//2 - 1)
# All functions return w aligned to this centered k-grid.
# Default normalization: sum(w) = 1 (L1).
# ============================================================

def kgrid(N: int) -> np.ndarray:
    """Centered integer frequency grid: k = -(N//2), ..., (N//2 - 1)."""
    return -(N // 2) + np.arange(N)

def normalize_l1(w: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """Normalize weights so sum(w)=1 (safe)."""
    s = np.sum(w)
    if np.abs(s) < eps:
        raise ValueError("Weight vector sums to ~0; cannot normalize.")
    return w / s

def enforce_real_if_close(w: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    """If imaginary part is tiny, drop it (helps keep things clean)."""
    if np.iscomplexobj(w) and np.max(np.abs(np.imag(w))) < tol:
        return np.real(w)
    return w

# ------------------------------------------------------------
# Dirichlet / rectangular window (uniform on kept band)
# ------------------------------------------------------------
def w_dirichlet(N: int) -> np.ndarray:
    """Uniform weights on I_N (rectangular window)."""
    return np.ones(N, dtype=float) / N

# ------------------------------------------------------------
# Fejér / Cesàro (triangular multiplier; real, even, >=0)
# ------------------------------------------------------------
def w_fejer(N: int) -> np.ndarray:
    """
    Fejér (Cesàro) weights on centered grid.
    Real, nonnegative, even, sum=1.
    """
    k = kgrid(N).astype(float)
    K = (N // 2) - 1
    if K <= 0:
        return w_dirichlet(N)
    w = np.maximum(0.0, 1.0 - (np.abs(k) / (K + 1.0)))
    return normalize_l1(w)


# ------------------------------------------------------------
# Jackson-like (stronger damping): (Fejér)^p, renormalized
# ------------------------------------------------------------
def w_jackson(N: int, p: int = 2) -> np.ndarray:
    """
    Jackson-like weights: (Fejér triangle)^p, renormalized.
    p=1 -> Fejér, p>1 -> stronger localization.
    """
    if p < 1:
        raise ValueError("p must be >= 1")
    w = w_fejer(N) ** p
    return normalize_l1(w)

# ------------------------------------------------------------
# General admissible-weight construction w(k) from g(k/N)
# Optionally neighbor-average (as in some constructions).
# ------------------------------------------------------------
def w_from_g(N: int, g, average_neighbors: bool = True) -> np.ndarray:
    """
    Build weights from an admissible weight function g(z) supported on [-1/2, 1/2],
    sampled at z = k/N (k on centered grid). Optionally do neighbor-averaging.
    """
    k = kgrid(N).astype(float)
    z = k / float(N)  # approx in [-1/2, 1/2)
    w0 = np.asarray(g(z), dtype=float)
    w0 = np.maximum(w0, 0.0)

    if average_neighbors:
        w = 0.5 * (w0 + np.roll(w0, -1))
    else:
        w = w0

    return normalize_l1(w)

# ------------------------------------------------------------
# B-spline-like family (compact support; beta controls smoothness)
# Proxy: g_beta(z) ∝ (1/4 - z^2)^(beta-1) on |z|<=1/2
# ------------------------------------------------------------
def w_bspline_like(N: int, beta: int = 2) -> np.ndarray:
    """
    Compactly-supported B-spline-like weights on z=k/N:
      g_beta(z) ∝ (1/4 - z^2)^(beta-1) for |z|<=1/2, else 0.
    beta=1 -> flatter
    beta=2 -> hat-ish
    higher -> smoother/more concentrated
    """
    if beta < 1:
        raise ValueError("beta must be >= 1")

    def g(z):
        base = 0.25 - z**2
        return np.where(base > 0, base ** (beta - 1), 0.0)

    return w_from_g(N, g, average_neighbors=True)

# ------------------------------------------------------------
# Sobolev-type weights (matches your sobg idea; as multiplier)
# g(z) ∝ (1/4 - z^2)^b / (gamma + |z|^(2a)) on |z|<=1/2
# ------------------------------------------------------------

def w_sobolev(N: int, a: float = 1.0, gamma: float = 1.0, norm: str = "K0") -> np.ndarray:
    """
    Sobolev-type frequency weights (CENTERED order):
      omega_k^{-1} = 1 + gamma * (2π |k|)^(2a)
      omega_k      = 1 / (1 + gamma * (2π |k|)^(2a))

    norm:
      - "none": no normalization
      - "dc":   enforce omega_{k=0} = 1  (already true here)
      - "K0":   enforce K(0)=sum_k omega_k = 1  (paper's stability assumption)
    """
    if a <= 0:
        raise ValueError("a must be > 0")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    k = kgrid(N).astype(float)
    w = 1.0 / (1.0 + gamma * (2.0 * np.pi * np.abs(k)) ** (2.0 * a))

    if norm == "none":
        return w
    if norm == "dc":
        return w / w[N // 2]
    if norm == "K0":
        return w / np.sum(w)
    raise ValueError("norm must be 'none', 'dc', or 'K0'")
# ------------------------------------------------------------
# Gaussian window in k
# ------------------------------------------------------------
def w_gaussian(N: int, sigma: float = 0.25) -> np.ndarray:
    """
    Gaussian multiplier in centered k:
      w_k ∝ exp(-(k/(sigma*K))^2), K≈N/2.
    sigma ~ 0.2-0.6 controls width (relative to half-band).
    """
    k = kgrid(N).astype(float)
    K = max(1.0, (N // 2) - 1.0)
    x = k / (sigma * K)
    w = np.exp(-(x**2))
    return normalize_l1(w)

# ------------------------------------------------------------
# Raised-cosine (Tukey/Hann-like) taper over k
# ------------------------------------------------------------
def w_raised_cosine(N: int, alpha: float = 1.0) -> np.ndarray:
    """
    Raised-cosine taper over centered k.
    alpha=1.0 -> full Hann-like taper
    alpha<1 -> flatter top (Tukey-like)
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")
    k = kgrid(N).astype(float)
    K = max(1.0, (N // 2) - 1.0)
    u = np.abs(k) / K  # in [0, ~1]
    w = np.zeros(N, dtype=float)

    # Flat region
    flat = u <= (1 - alpha)
    w[flat] = 1.0

    # Cosine taper region
    taper = (u > (1 - alpha)) & (u <= 1.0)
    # map u from (1-alpha, 1) to (0, pi)
    v = (u[taper] - (1 - alpha)) / alpha
    w[taper] = 0.5 * (1 + np.cos(np.pi * v))

    return normalize_l1(w)


# Convenience: dictionary of kernel factories (optional)
KERNELS = {
    "dirichlet": w_dirichlet,
    "fejer": w_fejer,
    "jackson": w_jackson,
    "gaussian": w_gaussian,
    "raised_cosine": w_raised_cosine,
    "bspline_like": w_bspline_like,
    "sobolev": w_sobolev
}


def infft(x, y, N, AhA=None, w=None, return_adjoint=False, approx=False, gpu=False):

    if x is None:
        raise ValueError("ERROR: No grid space, x, is specified.")
    if y is None:
        raise ValueError("ERROR: No amplitude, y, is specified.")
    if w is None:
        w = np.ones(N) / N

    if AhA is None and (approx is False):
        A = ndft_mat(x, N)
        AhA = A.H @ A

    use_gpu, nufft = _pick_nufft_backend(gpu=gpu)

    if not use_gpu:
        if approx is False:
            L, U = lu(AhA, permute_l=True)
            M = (np.diag(w)
                 - np.diag(w) @ L
                 @ np.linalg.pinv(np.eye(N) + U @ np.diag(w) @ L)
                 @ U @ np.diag(w))
            fk = nufft.nufft1d1(x, y, N, isign=-1) @ M
        else:
            fk = (nufft.nufft1d1(x, y, N, isign=-1) @ np.diag(w)) @ np.linalg.pinv(
                len(x) * np.diag(w) + np.eye(N)
            )

        if return_adjoint:
            fj = np.real(nufft.nufft1d2(x, fk, isign=+1))
            res_abs = np.sum(np.abs(y - fj) ** 2)
            res_rel = res_abs / np.sum(y ** 2)
        else:
            fj = res_abs = res_rel = None

        return fk, fj, res_abs, res_rel

    import torch
    device = _torch_device(gpu=True)

    x_t = _to_torch(x, device=device, dtype=torch.float64)
    y_dtype = torch.complex128 if (np.iscomplexobj(y) or torch.is_complex(_to_torch(y, device))) else torch.float64
    y_t = _to_torch(y, device=device, dtype=y_dtype)
    w_t = _to_torch(w, device=device, dtype=y_dtype)

    try:
        Fy = nufft.nufft1d1(x_t, y_t, N, isign=-1)
    except TypeError as e:
        # This is the exact failure mode you saw with NumPy, but now for Torch:
        raise TypeError(
            "cuFINUFFT did not accept Torch CUDA tensors. "
            "This usually means your cuFINUFFT build only supports arrays with __cuda_array_interface__ "
        ) from e

    # Ensure Fy is a torch tensor (depending on wrapper, it may return something else)
    Fy_t = Fy if torch.is_tensor(Fy) else torch.as_tensor(Fy, device=device)

    if approx is False:
        # LU of AhA on GPU
        AhA_t = _to_torch(AhA, device, torch.complex128)
        Nloc = AhA_t.shape[0]
        I = torch.eye(Nloc, dtype=torch.complex128, device=device)
        W = torch.diag(w_t)

        # Torch LU: P @ AhA = L @ U
        LUfac, piv = torch.linalg.lu_factor(AhA_t)
        P, L, U = torch.lu_unpack(LUfac, piv)
        Lp = P @ L

        X = I + (U @ W @ Lp)
        Xpinv = torch.linalg.pinv(X)
        M = W - (W @ Lp @ Xpinv @ U @ W)

        fk_t = Fy_t @ M

    else:
        # Approx branch, but keep it GPU-safe (no NumPy mixing)
        invdiag = 1.0 / (len(x_t) * w_t + 1.0)  # diagonal inverse
        fk_t = (Fy_t * w_t) * invdiag

    fk = fk_t.detach().cpu().numpy()

    if return_adjoint:
        # GPU adjoint if available through same backend
        fj_t = nufft.nufft1d2(x_t, fk_t, isign=+1)
        fj = torch.real(fj_t).detach().cpu().numpy()

        y_cpu = np.asarray(y)
        res_abs = np.sum(np.abs(y_cpu - fj) ** 2)
        res_rel = res_abs / np.sum(y_cpu ** 2)
    else:
        fj = res_abs = res_rel = None

    return fk, fj, res_abs, res_rel

def ndft_mat_nd(spatial_points, num_frequencies_per_dim):
    """
    Constructs the non-equispaced discrete Fourier transform (NDFT) matrix for N dimensions using matmul.

    C API is under sym_matrix.c in the core directory. This is very slow, and not recommended.

    Parameters:
        spatial_points (np.ndarray): Spatial points, shape (M, D) or (M,) for 1D.
        num_frequencies_per_dim (int or list[int]): Number of frequency points per dimension.
                                                    Can be an integer (1D) or list for N-D.

    Returns:
        np.ndarray: Transformation matrix A of shape (M, total_frequencies).
    """
    # Ensure spatial_points is a numpy array
    spatial_points = np.asarray(spatial_points)

    # Handle 1D case: If num_frequencies_per_dim is an integer, convert it to a list
    if isinstance(num_frequencies_per_dim, int):
        num_frequencies_per_dim = [num_frequencies_per_dim]

    # Handle the case where spatial_points is 1D (reshape to M x 1 for N-D compatibility)
    if spatial_points.ndim == 1:
        spatial_points = spatial_points[:, np.newaxis]  # Shape (M, 1)

    # Extract dimensions
    M, D = spatial_points.shape  # M: Number of spatial points, D: Dimensions
    if len(num_frequencies_per_dim) != D:
        raise ValueError("Length of num_frequencies_per_dim must match the dimensionality of spatial_points.")

    # Generate frequency points for each dimension
    frequency_grids = [-(N // 2) + np.arange(N) for N in num_frequencies_per_dim]
    if D == 1:  # Special case for 1D
        frequency_points = frequency_grids[0][:, np.newaxis]  # Shape (N, 1)
    else:
        frequency_points = np.array(np.meshgrid(*frequency_grids, indexing="ij")).reshape(D, -1).T  # Shape (total_frequencies, D)
    
    # Compute the transformation matrix
    # Outer product in N dimensions -> dot product between spatial and frequency points using matmul
    phase_matrix = 2j * np.pi * np.matmul(spatial_points, frequency_points.T)  # Shape (M, total_frequencies)
    transformation_matrix = np.exp(phase_matrix)
    
    return transformation_matrix

def infft_2d(data, N, AhA=None, w=None):
    """
    Perform a 2D inverse non-uniform FFT (INFFT) with operations applied along columns first.
    
    Parameters:
        data: np.ndarray
            2D input data array.
        N: int
            Length of the FFT.
        AhA: Optional
            Precomputed matrix or value for the inverse operation.
        w: np.ndarray, optional
            Weight vector for the given axis.
    
    Returns:
        tuple:
            - np.ndarray: The transformed data in 2D.
            - list: Mean values for each column.
    """
    t = np.linspace(-0.5, 0.5, data.shape[0], endpoint=False)
    h_k = -(N // 2) + np.arange(N)

    ftot_list = []
    mtot_list = []

    # Step 1: Column-wise INFFT
    for jj in range(data.shape[1]):
        idx = ~np.isnan(data[:, jj])  # Identify valid (non-NaN) values
        if np.sum(idx) == 0:
            ftot_list.append(np.zeros(N, dtype=complex))
            mtot_list.append(0)
            continue

        AAh = compute_sym_matrix_optimized(t[idx], h_k)

        if np.sum(idx) % 2 != 0:
            idx[np.where(idx)[0][-1]] = False  # Adjust to even number of samples if needed

        mn = np.mean(data[idx, jj])
        ftot, _, _, _ = infft(t[idx], data[idx, jj] - mn, N=N, AhA=AAh, w=w)

        ftot_list.append(ftot)
        mtot_list.append(mn)

    ftot = np.array(ftot_list).T
    mtot = np.array(mtot_list)

    # Step 2: Row-wise FFT
    result = np.fft.fft(ftot, axis=1)

    return result, mtot


def adjoint_transform_2d(transformed_data, mtot, data_shape):
    """
    Perform the adjoint of the 2D forward transform to reconstruct the original data, with operations along rows.
    
    Parameters:
        transformed_data: np.ndarray
            2D transformed data array.
        mtot: list
            Mean values for each column from the forward transform.
        data_shape: tuple
            Shape of the original data (to restore NaNs).
    
    Returns:
        np.ndarray
            Reconstructed data including NaNs in their original locations.
    """
    t = np.linspace(-0.5, 0.5, data_shape[0], endpoint=False)

    reconstructed_data = np.full(data_shape, np.nan, dtype=np.float64)  # Initialize with NaNs

    # Step 1: Inverse row-wise FFT
    iftot = np.fft.ifft(transformed_data, axis=1)

    # Step 2: Column-wise adjoint transform
    for jj in range(data_shape[1]):
        # Perform the adjoint operation
        #adjoint_result = adjoint(t, iftot[:, jj])
        adjoint_result = finufft.nufft1d2(t, iftot[:,jj], isign=+1)

        # Restore mean and NaNs
        reconstructed_column = adjoint_result + mtot[jj]
        reconstructed_data[:, jj] = np.abs(reconstructed_column)

    return reconstructed_data

def compute_sym_matrix_optimized(f_j, h_k):
    """
    Compute the inner product matrix when h_k is equally spaced.
    That is, compute the Toeplitz matrix A with
        A[k1, k2] = sum_j exp(2πi * f_j * (h_k[k2] - h_k[k1])),
    where h_k[k] = h0 + k*d.
    """
    f_j = np.asarray(f_j, dtype=float)
    h_k = np.asarray(h_k, dtype=float)
    N = h_k.size

    # Verify that h_k is equally spaced.
    d = h_k[1] - h_k[0]
    if not np.allclose(np.diff(h_k), d):
        raise ValueError("h_k must be equally spaced for a Toeplitz structure.")

    # Compute the unique values for the first column.
    # The lag for entry (0, k) is h_k[k] - h_k[0] = d * k.
    lags = d * np.arange(N)
    #col = np.array([np.sum(np.exp(-2 * np.pi * 1j * f_j * lag)) for lag in lags])
    col = np.array([np.sum(np.exp(1j * f_j * lag)) for lag in lags])


    # For a Toeplitz matrix, the entry A[i,j] depends only on (j-i).
    # Since A[0,j] is given by 'col', we build the full matrix.
    A = toeplitz(col)
    
    # For numerical precision, enforce that the diagonal is exactly M.
    # Here, on the diagonal, lag=0 so exp(0)=1, and sum_j 1 = len(f_j).
    np.fill_diagonal(A, len(f_j))
    
    return A

