import os, sys, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths inside the uploaded tarball extraction
# extract_dir="/mnt/data/release_extract"
# pkg_root=os.path.join(extract_dir, "iinfft")
# data_path=os.path.join(pkg_root, "data", "modified_image.npy")

# # Make sure we can import the local package without installing
# sys.path.insert(0, extract_dir)

from iinfft.iinfft import infft_2d, adjoint_transform_2d, is_gpu, w_sobolev

data_path = "iinfft/data/modified_image.npy"

def bench_backend(data, Ns,w=None, reps=3, gpu=False, label="cpu"):
    rows=[]
    # warmup (tiny) to init libs
    N0=Ns[0]
    _=infft_2d(data, N0, w=w[0], gpu=gpu)
    for i, N in enumerate(Ns):
        for r in range(reps):
            t0=time.perf_counter()
            F, mtot = infft_2d(data, N=N, w=w[i], gpu=gpu)
            t1=time.perf_counter()
            Xhat = adjoint_transform_2d(F, mtot, data.shape, gpu=gpu)
            t2=time.perf_counter()

            mask=~np.isnan(data)
            err = Xhat[mask] - data[mask]
            rmse = float(np.sqrt(np.mean(err*err)))

            # MAPE on nonzero entries
            denom = np.abs(data[mask])
            nz = denom > 1e-12
            mape = float(np.mean(np.abs(err[nz]) / denom[nz])) if np.any(nz) else float("nan")

            rows.append({
                "backend": label,
                "gpu": bool(gpu),
                "N": int(N),
                "rep": int(r),
                "t_forward_s": float(t1-t0),
                "t_adjoint_s": float(t2-t1),
                "t_total_s": float(t2-t0),
                "rmse": rmse,
                "mape": mape,
            })
            print(f"[rep {r+1}/3] N={N:4d} | gpu={str(gpu):5s}")
    return pd.DataFrame(rows)

# Load data and take a manageable crop for this environment
X = np.load(data_path)
Xc = X.copy()  # deterministic crop; change as desired
Ns = [4, 8, 16, 32,64,128]
weights = [
    w_sobolev(N, a=0.5, b=3, gamma=1e-1)
    for N in Ns
]
reps = 3

dfs=[]
dfs.append(bench_backend(Xc, Ns, w=weights, reps=reps, gpu=False, label="cpu"))

gpu_available = is_gpu()
if gpu_available:
    dfs.append(bench_backend(Xc, Ns, w=weights, reps=reps, gpu=True, label="gpu"))
df = pd.concat(dfs, ignore_index=True)

# Aggregate
agg = (df.groupby(["backend","N"])
         .agg(t_total_mean=("t_total_s","mean"),
              t_total_std=("t_total_s","std"),
              t_fwd_mean=("t_forward_s","mean"),
              t_adj_mean=("t_adjoint_s","mean"),
              rmse_mean=("rmse","mean"),
              rmse_std=("rmse","std"),
              mape_mean=("mape","mean"))
         .reset_index())

# Save CSV + plot
out_csv="iinfft_2d_benchmark_results.csv"
out_png="iinfft_2d_cpu_gpu_benchmark.png"
df.to_csv(out_csv, index=False)

plt.figure()
for backend, sub in agg.groupby("backend"):
    plt.errorbar(sub["N"], sub["t_total_mean"], yerr=sub["t_total_std"], marker="o", capsize=3, label=backend)
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("Number of coefficients (N)")
plt.ylabel("End-to-end time (s): infft_2d + adjoint")
plt.title("2D dataset benchmark")
plt.legend()
plt.tight_layout()
plt.savefig(out_png, dpi=200)

out_png, out_csv, gpu_available, agg.head(10)