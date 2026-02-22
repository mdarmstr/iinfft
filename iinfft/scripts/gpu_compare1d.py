import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from iinfft.iinfft import (
    infft,
    compute_sym_matrix_optimized,
    w_sobolev,
    is_gpu,
)

data_path = "iinfft/data/T.Suelo.csv"

def bench_backend_1d(t_full, dat_full, Ns, weights, reps=3, gpu=False, label="cpu"):
    rows = []

    # mask missing values exactly like your example
    idx = dat_full != -9999
    t = t_full[idx].astype(np.float32)
    dat = dat_full[idx].astype(np.complex64)

    mn = np.mean(dat)
    y0 = dat - mn

    # Warmup
    N0 = Ns[0]
    h0 = -(N0 // 2) + np.arange(N0)
    AhA0 = compute_sym_matrix_optimized(t, h0,gpu=gpu)
    _ = infft(t, y0, N=N0, AhA=AhA0, w=weights[0], gpu=gpu, return_adjoint=True)

    for i, N in enumerate(Ns):
        w = weights[i]
        h_k = -(N // 2) + np.arange(N)

        # Build AhA once per N (keep comparable; and don’t rebuild inside reps)
        tA0 = time.perf_counter()
        AhA = compute_sym_matrix_optimized(t, h_k,gpu=gpu)
        tA1 = time.perf_counter()
        t_AhA = float(tA1 - tA0)

        for r in range(reps):
            print(f"[rep {r+1}/{reps}] N={N:4d} | gpu={str(gpu):5s}", flush=True)

            t0 = time.perf_counter()

            # IMPORTANT: return_adjoint=True to get reconstruction directly
            out = infft(t, y0, N=N, AhA=AhA, w=w, gpu=gpu, return_adjoint=True)

            t1 = time.perf_counter()

            # Be robust to your return signature
            # Expected: ftot, ytot, _, _  (like your usage with return_adjoint=False)
            ftot = out[0]
            yrec = out[1]  # reconstructed values at t (observed points)

            # add back mean so yrec is comparable to dat
            yrec = yrec + mn

            # Error metrics on observed points
            err = (yrec - dat)
            rmse = float(np.sqrt(np.mean(np.abs(err)**2)))

            denom = np.abs(dat)
            nz = denom > 1e-12
            mape = float(np.mean(np.abs(err[nz]) / denom[nz])) if np.any(nz) else float("nan")

            rows.append({
                "backend": label,
                "gpu": bool(gpu),
                "N": int(N),
                "rep": int(r),

                # timings
                "t_AhA_s": t_AhA,                 # computed once per N
                "t_infft_s": float(t1 - t0),      # per replicate (solve+adjoint)
                "t_total_s": float(t_AhA + (t1 - t0)),

                # quality
                "rmse": rmse,
                "mape": mape,
            })

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(data_path)
    Ln = df.shape[0]
    t = 2*np.pi*np.linspace(-0.5, 0.5, Ln, endpoint=False)

    # choose the same column style as you did (e.g. column 4)
    data_raw = df.iloc[:, 1:].to_numpy()
    dat = data_raw[:, 4].astype(np.complex64)

    Ns = [16, 32, 64, 128, 256, 512, 1024]
    weights = [w_sobolev(N, a=1, b=2, gamma=1e-1) for N in Ns]
    reps = 3

    dfs = []
    dfs.append(bench_backend_1d(t, dat, Ns, weights, reps=reps, gpu=False, label="cpu"))

    gpu_available = is_gpu()
    if gpu_available:
        dfs.append(bench_backend_1d(t, dat, Ns, weights, reps=reps, gpu=True, label="gpu"))

    res = pd.concat(dfs, ignore_index=True)

    agg = (res.groupby(["backend", "N"])
              .agg(t_infft_mean=("t_infft_s", "mean"),
                   t_infft_std=("t_infft_s", "std"),
                   t_total_mean=("t_total_s", "mean"),
                   t_total_std=("t_total_s", "std"),
                   rmse_mean=("rmse", "mean"),
                   rmse_std=("rmse", "std"),
                   mape_mean=("mape", "mean"))
              .reset_index())

    out_csv = "iinfft_1d_benchmark_results.csv"
    out_png = "iinfft_1d_cpu_gpu_benchmark.png"
    res.to_csv(out_csv, index=False)

    # Plot total time vs N
    plt.figure()
    for backend, sub in agg.groupby("backend"):
        plt.errorbar(sub["N"], sub["t_total_mean"], yerr=sub["t_total_std"],
                     marker="o", capsize=3, label=backend)

    plt.xscale("log", base=2)
    #plt.yscale("log")
    plt.xlabel("Number of coefficients (N)")
    plt.ylabel("Time (s): AhA build + infft(return_adjoint=True)")
    plt.title("1D dataset benchmark (3 replicates)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    print("Saved:", out_csv)
    print("Saved:", out_png)
    print("GPU available:", gpu_available)
    print(agg)

if __name__ == "__main__":
    main()