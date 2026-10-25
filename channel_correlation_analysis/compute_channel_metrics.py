"""
Physical channel metrics per cluster for LOS/NLOS characterisation.

Reads the calibrated CSI (`mob_csi`, 64 antennas x frames x 52 subcarriers) and
reports power-independent statistics that separate LOS from NLOS:
  eig1   : dominant-eigenvalue share of the spatial covariance R = E[h h^H]
  temp   : normalised correlation between consecutive frames
  freq   : normalised correlation between adjacent subcarriers
  K_dB   : Rician K-factor (moment method)
  tau_ns : RMS delay spread from the power-delay profile
  ampCV  : coefficient of variation of |h| across subcarriers

Usage:
  python compute_channel_metrics.py --data_dir ../tvt_open_source/channel_data
"""

import argparse
import os

import h5py
import numpy as np

CLUSTERS = ["LOS_cluster1", "LOS_cluster2", "LOS_cluster3", "LOS_cluster4",
            "NLOS_cluster1", "NLOS_cluster2", "NLOS_cluster3",
            "NLOS_cluster4", "NLOS_cluster5"]
WINDOW = 50           # frames per window for covariance and K-factor
SUBCARRIER_HZ = 312.5e3


def normalised_corr(a, b, axis):
    """rho = |a^H b| / (||a|| ||b||) along `axis`."""
    num = np.abs(np.sum(a.conj() * b, axis=axis))
    return num / (np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis))


def eigen_share(H, window=WINDOW, step=4 * WINDOW, sub_step=8):
    """Mean lambda_1 / sum(lambda) of the 64x64 spatial covariance over windows."""
    shares = []
    for s in range(0, len(H) - window, step):
        X = H[s:s + window]
        X = X / np.linalg.norm(X, axis=1, keepdims=True)  # remove power changes
        for k in range(0, X.shape[2], sub_step):
            R = X[:, :, k].T @ X[:, :, k].conj() / window
            ev = np.linalg.eigvalsh(R)
            shares.append(ev[-1] / ev.sum())
    return np.mean(shares)


def rician_k_db(H, window=WINDOW):
    """Moment-method K-factor: gamma = Var(|h|^2)/E[|h|^2]^2, K = sqrt(1-g)/(1-sqrt(1-g))."""
    P = np.abs(H) ** 2
    ks = []
    for s in range(0, len(H) - window, window):
        p = P[s:s + window]
        g = np.clip(p.var(0) / p.mean(0) ** 2, 1e-6, 1 - 1e-6)
        r = np.sqrt(1 - g)
        ks.append(np.median(r / (1 - r)))
    return 10 * np.log10(np.mean(ks))


def rms_delay_spread_ns(H, n_fft=64):
    pdp = (np.abs(np.fft.ifft(H, n=n_fft, axis=2)) ** 2).mean(1)
    pdp = np.roll(pdp, 8, axis=1)                        # keep the first tap off the edge
    tau = np.arange(n_fft) / (n_fft * SUBCARRIER_HZ) * 1e9
    m1 = (pdp * tau).sum(1) / pdp.sum(1)
    m2 = (pdp * tau ** 2).sum(1) / pdp.sum(1)
    return np.median(np.sqrt(np.maximum(m2 - m1 ** 2, 0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="../tvt_open_source/channel_data")
    args = p.parse_args()

    print(f"{'cluster':14s} {'eig1':>6s} {'temp':>6s} {'freq':>6s} "
          f"{'K_dB':>6s} {'tau_ns':>7s} {'ampCV':>6s}")
    for c in CLUSTERS:
        path = os.path.join(args.data_dir, f"processed_channe14_{c}.hdf5")
        with h5py.File(path, "r") as f:
            H = f["mob_csi"][:].transpose(1, 0, 2).astype(np.complex64)  # (T, 64, 52)

        temp = normalised_corr(H[:-1], H[1:], axis=1).mean()
        freq = normalised_corr(H[:, :, :-1], H[:, :, 1:], axis=1).mean()
        A = np.abs(H)
        amp_cv = (A.std(2) / A.mean(2)).mean()

        print(f"{c:14s} {eigen_share(H):6.3f} {temp:6.3f} {freq:6.3f} "
              f"{rician_k_db(H):6.1f} {rms_delay_spread_ns(H):7.1f} {amp_cv:6.3f}")


if __name__ == "__main__":
    main()
