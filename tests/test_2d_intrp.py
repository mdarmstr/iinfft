import numpy as np

from iinfft.iinfft import w_sobolev, infft_2d, adjoint_transform_2d

def test_2d_reconstruction_fills_nans():
    image = np.load("iinfft/data/modified_image.npy")
    nan_mask = np.isnan(image)

    N = 32  # smaller for CI speed
    w = w_sobolev(N, a=0.5, b=3.0, gamma=1e-1)

    transformed, mtot = infft_2d(image, N, w=w, gpu=False)
    recon = adjoint_transform_2d(transformed, mtot, data_shape=image.shape, gpu=False)

    assert transformed.shape[0] == N
    assert transformed.shape[1] == image.shape[1]
    assert recon.shape == image.shape

    # recon should be finite everywhere (your adjoint returns numeric values)
    assert np.isfinite(recon).all()

    # After interpolation, NaNs should be gone
    interpolated = image.copy()
    interpolated[nan_mask] = recon[nan_mask]
    assert not np.isnan(interpolated).any()