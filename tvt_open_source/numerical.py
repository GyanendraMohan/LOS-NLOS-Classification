#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical Simulations
@author: Xu Du
"""
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
from lib import dsp_lib
import argparse

LIGHT_SPEED = 299792458.0


def generate_NLOS_channel(
    M_x,
    M_y,
    sin_x_thetas,
    sin_y_thetas,
    Delta_x=0.5,
    Delta_y=0.5,
    beta_nlos=0.25,
):
    h = np.zeros((int(M_x * M_y)), dtype=complex)

    for sin_theta_x, sin_theta_y in zip(sin_x_thetas, sin_y_thetas):
        random_phase = dsp_lib.get_random_phis_coeff()
        antenna_idxs = []
        for antenna_idx in range(int(M_x) * int(M_y)):
            antenna_idx_x = antenna_idx // M_x
            antenna_idx_y = antenna_idx % M_x
            antenna_idxs.append((antenna_idx_x, antenna_idx_y))
            x_factor = np.exp(
                -1.0j * Delta_x * antenna_idx_x * 2 * np.pi * sin_theta_x
            )
            y_factor = np.exp(
                -1.0j * Delta_y * antenna_idx_y * 2 * np.pi * sin_theta_y
            )
            h[antenna_idx] += beta_nlos * random_phase * x_factor * y_factor
    return h


def get_case_to_stats(
    M_xs,
    M_ys,
    alpha_x,
    alpha_y,
    num_paths,
    Delta_x=0.5,
    Delta_y=0.5,
    num_samples=20000,
    verbose=False,
    use_cache=True,
):
    def _hahs_list(Ms):
        import hashlib

        return str(
            int(hashlib.sha1(str(Ms).encode("utf-8")).hexdigest(), 16)
            % (10 ** 8)
        )

    def _gen_cache_name(names):
        return (
            "data/_numerical_cache_"
            + "_".join([str(n) for n in names])
            + ".hdf5"
        )

    def _write_cache(data, names):
        dd.io.save(_gen_cache_name(names), data)

    def _get_cache(names):
        return dd.io.load(_gen_cache_name(names))

    names = [
        _hahs_list(M_xs),
        _hahs_list(M_ys),
        alpha_x,
        alpha_y,
        num_paths,
        Delta_x,
        Delta_y,
        num_samples,
    ]

    if use_cache:
        try:
            M_to_mean = _get_cache(["M_to_mean"] + names)
            M_to_var = _get_cache(["M_to_var"] + names)
            M_to_samples = _get_cache(["M_to_samples"] + names)
            if not verbose:
                return M_to_mean, M_to_var
            else:
                return M_to_mean, M_to_var, M_to_samples
        except BaseException:
            print("Failed to obtain cache!")
            M_to_mean = {}
            M_to_var = {}
            M_to_samples = {}
    else:
        M_to_mean = {}
        M_to_var = {}
        M_to_samples = {}

    for M_x, M_y in zip(M_xs, M_ys):
        print("processing Mx = {}, M_y = {}".format(M_x, M_y))
        samples = [0] * num_samples
        for idx in range(num_samples):
            sinx_thetas_0 = []
            siny_thetas_0 = []
            sinx_thetas_1 = []
            siny_thetas_1 = []
            for _ in range(num_paths):
                sinx_theta0, sinx_theta1 = dsp_lib.random_sin_theta(alpha_x)
                sinx_thetas_0.append(sinx_theta0)
                sinx_thetas_1.append(sinx_theta1)

                siny_theta0, siny_theta1 = dsp_lib.random_sin_theta(alpha_y)
                siny_thetas_0.append(siny_theta0)
                siny_thetas_1.append(siny_theta1)
            h1 = generate_NLOS_channel(
                M_x,
                M_y,
                sinx_thetas_0,
                siny_thetas_0,
                Delta_x=Delta_x,
                Delta_y=Delta_y,
                beta_nlos=1.0 / np.sqrt(num_paths),
            )
            h2 = generate_NLOS_channel(
                M_x,
                M_y,
                sinx_thetas_1,
                siny_thetas_1,
                Delta_x=Delta_x,
                Delta_y=Delta_y,
                beta_nlos=1.0 / np.sqrt(num_paths),
            )
            samples[idx] = dsp_lib.compute_vector_correlation(h1, h2)

        M_to_mean[M_x, M_y] = np.mean(samples).astype(np.float64)
        # abs_samples = [np.abs(sample) for sample in samples]
        M_to_var[M_x, M_y] = np.var(samples).astype(np.float64)
        if verbose:
            M_to_samples[M_x, M_y] = samples

    _write_cache(M_to_mean, ["M_to_mean"] + names)
    _write_cache(M_to_var, ["M_to_var"] + names)
    _write_cache(M_to_samples, ["M_to_samples"] + names)

    if not verbose:
        return M_to_mean, M_to_var
    else:
        return M_to_mean, M_to_var, M_to_samples


def plot_2d_square_var(
    array_label,
    alpha_xys,
    fig_idx,
    M_xs,
    M_ys,
    Delta_x=0.5,
    Delta_y=0.5,
    num_paths=1,
    num_samples=10 ** 3,
    use_cache=True,
    fig_prefix="fig_7",
):
    for alpha_x, alpha_y in alpha_xys:
        print("--" * 10)
        print("Porcessing alpha_x={} alpha_y={}".format(alpha_x, alpha_y))
        M_to_mean, M_to_var = get_case_to_stats(
            M_xs,
            M_ys,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
            num_paths=num_paths,
            Delta_x=Delta_x,
            Delta_y=Delta_y,
            num_samples=num_samples,
            use_cache=use_cache,
        )
        plt.figure(fig_idx)
        Ms = [M_x * M_y for M_x, M_y in zip(M_xs, M_ys)]
        sim_res_data = [
            np.sqrt(M_to_var[M_x, M_y]) for M_x, M_y in zip(M_xs, M_ys)
        ]
        plt.plot(
            Ms,
            sim_res_data,
            color="seagreen",
            ls="-",
            lw=1.5,
            marker="x",
            markersize=10,
            markerfacecolor="None",
            markeredgecolor="seagreen",
            alpha=0.7,
        )
        theory_res_data = [
            np.sqrt(
                dsp_lib.theory_res(
                    M_x=M_x,
                    alpha_x=alpha_x,
                    Delta_x=Delta_x,
                    M_y=M_y,
                    alpha_y=alpha_y,
                    Delta_y=Delta_y,
                    num_paths=num_paths,
                )[1]
            )
            for M_x, M_y in zip(M_xs, M_ys)
        ]
        print("--" * 10)
        print(Ms)
        print(
            [
                theory_res_data[idx] / sim_res_data[idx]
                for idx in range(len(sim_res_data))
            ]
        )
        plt.plot(
            Ms,
            theory_res_data,
            ls="-",
            color="r",
            lw=1.5,
            marker="o",
            markersize=10,
            markerfacecolor="None",
            markeredgecolor="r",
            alpha=0.6,
        )
        plt.plot(
            [M_x * M_y for M_x, M_y in zip(M_xs, M_ys) if M_x * M_y > 50],
            [
                np.sqrt(
                    dsp_lib.theory_res_asy(
                        M_x=M_x,
                        alpha_x=alpha_x,
                        Delta_x=Delta_x,
                        M_y=M_y,
                        alpha_y=alpha_y,
                        Delta_y=Delta_y,
                        num_paths=num_paths,
                    )[1]
                )
                for M_x, M_y in zip(M_xs, M_ys)
                if M_x * M_y > 50
            ],
            ls="--",
            color="blue",
            lw=1.5,
            marker="^",
            markersize=10,
            markerfacecolor="None",
            markeredgecolor="blue",
        )
    plt.plot(Ms, [1 / np.sqrt(float(M)) for M in Ms], "black", marker=".")
    lines = [
        Line2D(
            [0],
            [0],
            color="blue",
            linewidth=2,
            linestyle="--",
            marker="^",
            markersize=10,
            markerfacecolor="None",
            markeredgecolor="blue",
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=2,
            linestyle="-",
            markersize=10,
            marker="o",
            markerfacecolor="None",
            markeredgecolor="red",
        ),
        Line2D(
            [0],
            [0],
            color="seagreen",
            linewidth=2,
            linestyle="-",
            markersize=10,
            marker="x",
            markerfacecolor="None",
            markeredgecolor="seagreen",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2,
            linestyle="-",
            marker=".",
            markerfacecolor="black",
            markeredgecolor="black",
        ),
    ]
    labels = [
        "Proposition 2",
        "Proposition 1",
        "Simulation",
        "i.i.d. Rayleigh Fading",
    ]
    plt.legend(lines, labels, loc="upper right")
    plt.xlabel("Array Size, M", fontsize=25)
    plt.ylabel("User Channel Correlation", fontsize=25)
    plt.axis([0, 625, -0.02, 1.0])
    plt.grid(True, alpha=0.3)
    name_pattern = "plots/{fig_prefix}_ray_analysis_cor_{array_label}_var_num_paths_{num_paths}_{Delta_x}_{Delta_y}"
    plt.savefig(
        name_pattern.format(
            fig_prefix=fig_prefix,
            array_label=array_label,
            num_paths=num_paths,
            Delta_x=Delta_x,
            Delta_y=Delta_y,
        ).replace(".", "_")
        + ".pdf"
    )
    plt.close()


def plot_2d_los_square_mean(Delta_x=0.5, Delta_y=0.5, num_samples=10 ** 3):
    M_xs = [m for m in range(1, 102, 10)]
    M_ys = [m for m in M_xs]
    for alpha_x, alpha_y in [  # 0.9,
        (0.6, 0.6),
        (0.4, 0.4),
        (0.2, 0.2),
        (0.1, 0.1),
    ]:
        print("--" * 10)
        print("Porcessing alpha_x={} alpha_y={}".format(alpha_x, alpha_y))
        M_to_mean, M_to_var = get_case_to_stats(
            M_xs,
            M_ys,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
            num_paths=1,
            Delta_x=Delta_x,
            Delta_y=Delta_y,
            num_samples=num_samples,
        )
        plt.figure(0)
        Ms = [M_x * M_y for M_x, M_y in zip(M_xs, M_ys)]
        sim_res_data = [M_to_mean[M_x, M_y] for M_x, M_y in zip(M_xs, M_ys)]
        plt.plot(
            Ms,
            sim_res_data,
            color="seagreen",
            ls="-",
            lw=1.5,
            marker="x",
            markersize=10,
            markerfacecolor="None",
            markeredgecolor="seagreen",
            alpha=0.7,
        )
        theory_res_data = [0 for M_x, M_y in zip(M_xs, M_ys)]
        print(Ms)
        print(sim_res_data)
        print(theory_res_data)
        plt.plot(
            Ms,
            theory_res_data,
            ls="-",
            color="r",
            lw=1.5,
            marker="o",
            markersize=10,
            markerfacecolor="None",
            markeredgecolor="r",
            alpha=0.6,
        )
        plt.plot(
            [M_x * M_y for M_x, M_y in zip(M_xs, M_ys) if M_x * M_y > 50],
            [0 for M_x, M_y in zip(M_xs, M_ys) if M_x * M_y > 50],
            ls="--",
            color="blue",
            lw=1.5,
            marker="^",
            markersize=10,
            markerfacecolor="None",
            markeredgecolor="blue",
        )
    lines = [
        Line2D(
            [0],
            [0],
            color="blue",
            linewidth=2,
            linestyle="--",
            marker="^",
            markersize=10,
            markerfacecolor="None",
            markeredgecolor="blue",
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=2,
            linestyle="-",
            markersize=10,
            marker="o",
            markerfacecolor="None",
            markeredgecolor="red",
        ),
        Line2D(
            [0],
            [0],
            color="seagreen",
            linewidth=2,
            linestyle="-",
            markersize=10,
            marker="x",
            markerfacecolor="None",
            markeredgecolor="seagreen",
        ),
    ]
    labels = ["Proposition 2", "Proposition 1", "Simulation"]
    plt.legend(lines, labels, loc="upper right")
    plt.xlabel("Array Size, M", fontsize=25)
    plt.ylabel(
        r"Var$[\frac{1}{M} \mathbf{h}_{k}\mathbf{h}_{k^{'}}]$", fontsize=25
    )
    plt.axis([0, 250, -0.02, 1.0])
    plt.grid(True, alpha=0.3)
    plt.savefig(
        "plots/ray_analysis_los_var_square_mean_{Delta_x}_{Delta_y}".format(
            Delta_x=Delta_x, Delta_y=Delta_y
        ).replace(".", "_")
        + ".pdf"
    )


def plot_2d_var_D_vs_alpha(
    array_label,
    alpha_xy_and_D_xys,
    fig_idx,
    M_xs,
    M_ys,
    num_paths=1,
    num_samples=10 ** 3,
    use_cache=True,
):
    for alpha_x, alpha_y, Delta_x, Delta_y in alpha_xy_and_D_xys:
        print("--" * 10)
        print("Porcessing alpha_x={} alpha_y={}".format(alpha_x, alpha_y))
        M_to_mean, M_to_var = get_case_to_stats(
            M_xs,
            M_ys,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
            num_paths=num_paths,
            Delta_x=Delta_x,
            Delta_y=Delta_y,
            num_samples=num_samples,
            use_cache=use_cache,
        )
        plt.figure(fig_idx)
        Ms = [M_x * M_y for M_x, M_y in zip(M_xs, M_ys)]
        sim_res_data = [
            np.sqrt(M_to_var[M_x, M_y]) for M_x, M_y in zip(M_xs, M_ys)
        ]
        plt.plot(
            Ms,
            sim_res_data,
            color="seagreen",
            ls="-",
            lw=1.5,
            marker="x",
            markersize=10,
            markerfacecolor="None",
            markeredgecolor="seagreen",
            alpha=0.7,
        )
        theory_res_data = [
            np.sqrt(
                dsp_lib.theory_res(
                    M_x=M_x,
                    alpha_x=alpha_x,
                    Delta_x=Delta_x,
                    M_y=M_y,
                    alpha_y=alpha_y,
                    Delta_y=Delta_y,
                    num_paths=num_paths,
                )[1]
            )
            for M_x, M_y in zip(M_xs, M_ys)
        ]
        print("--" * 10)
        print(Ms)
        print(
            [
                theory_res_data[idx] / sim_res_data[idx]
                for idx in range(len(sim_res_data))
            ]
        )
        plt.plot(
            Ms,
            theory_res_data,
            ls="-",
            color="r",
            lw=1.5,
            marker="o",
            markersize=10,
            markerfacecolor="None",
            markeredgecolor="r",
            alpha=0.6,
        )
        plt.plot(
            [M_x * M_y for M_x, M_y in zip(M_xs, M_ys) if M_x * M_y > 50],
            [
                np.sqrt(
                    dsp_lib.theory_res_asy(
                        M_x=M_x,
                        alpha_x=alpha_x,
                        Delta_x=Delta_x,
                        M_y=M_y,
                        alpha_y=alpha_y,
                        Delta_y=Delta_y,
                        num_paths=num_paths,
                    )[1]
                )
                for M_x, M_y in zip(M_xs, M_ys)
                if M_x * M_y > 50
            ],
            ls="--",
            color="blue",
            lw=1.5,
            marker="^",
            markersize=10,
            markerfacecolor="None",
            markeredgecolor="blue",
        )
    plt.plot(Ms, [1 / np.sqrt(float(M)) for M in Ms], "black", marker=".")
    lines = [
        Line2D(
            [0],
            [0],
            color="blue",
            linewidth=2,
            linestyle="--",
            marker="^",
            markersize=10,
            markerfacecolor="None",
            markeredgecolor="blue",
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=2,
            linestyle="-",
            markersize=10,
            marker="o",
            markerfacecolor="None",
            markeredgecolor="red",
        ),
        Line2D(
            [0],
            [0],
            color="seagreen",
            linewidth=2,
            linestyle="-",
            markersize=10,
            marker="x",
            markerfacecolor="None",
            markeredgecolor="seagreen",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2,
            linestyle="-",
            marker=".",
            markerfacecolor="black",
            markeredgecolor="black",
        ),
    ]
    labels = [
        "Proposition 2",
        "Proposition 1",
        "Simulation",
        "i.i.d. Rayleigh Fading",
    ]
    plt.legend(lines, labels, loc="upper right")
    plt.xlabel("Array Size, M", fontsize=25)
    plt.ylabel("User Channel Correlation", fontsize=25)
    plt.axis([0, 625, -0.02, 1.0])
    plt.grid(True, alpha=0.3)
    name_pattern = (
        "plots/fig_9_ray_analysis_cor_{array_label}_var_num_paths_{num_paths}"
    )
    plt.savefig(
        name_pattern.format(
            array_label=array_label,
            num_paths=num_paths,
            Delta_x=Delta_x,
            Delta_y=Delta_y,
        ).replace(".", "_")
        + ".pdf"
    )
    plt.close()


def plot_2d_var_subcarrier_variance(
    array_label,
    alpha_xys,
    fig_idx,
    M_xs,
    M_ys,
    num_paths=1,
    num_samples=10 ** 3,
    use_cache=True,
):
    central_frequency = 2.437 * 1000000000  # 2.4 GHz
    sub_car_idxs = [idx for idx in range(-26, 27, 1) if idx]
    sub_frequency_delta = 10000000 / 26.0  # 10 MHz/ 26
    for alpha_x, alpha_y in alpha_xys:
        M_to_var_sim = {}
        M_to_var_theory = {}
        M_to_var_theory_asy = {}
        for sub_idx in sub_car_idxs:
            print("--" * 10)
            print(
                "Porcessing alpha_x={} alpha_y={}, sub_idx={}".format(
                    alpha_x, alpha_y, sub_idx
                )
            )
            d = 0.0635
            freq = central_frequency + sub_frequency_delta * sub_idx
            Delta_x = d / (LIGHT_SPEED / freq)
            Delta_y = Delta_x
            print(Delta_x, Delta_y)
            _, M_to_var_per_sub = get_case_to_stats(
                M_xs,
                M_ys,
                alpha_x=alpha_x,
                alpha_y=alpha_y,
                num_paths=num_paths,
                Delta_x=Delta_x,
                Delta_y=Delta_y,
                num_samples=num_samples,
                use_cache=use_cache,
            )
            M_to_var_sim[sub_idx] = M_to_var_per_sub
            M_to_var_theory[sub_idx] = {
                (M_x, M_y): np.sqrt(
                    dsp_lib.theory_res(
                        M_x=M_x,
                        alpha_x=alpha_x,
                        Delta_x=Delta_x,
                        M_y=M_y,
                        alpha_y=alpha_y,
                        Delta_y=Delta_y,
                        num_paths=num_paths,
                    )[1]
                )
                for M_x, M_y in zip(M_xs, M_ys)
            }
            M_to_var_theory_asy[sub_idx] = {
                (M_x, M_y): np.sqrt(
                    dsp_lib.theory_res_asy(
                        M_x=M_x,
                        alpha_x=alpha_x,
                        Delta_x=Delta_x,
                        M_y=M_y,
                        alpha_y=alpha_y,
                        Delta_y=Delta_y,
                        num_paths=num_paths,
                    )[1]
                )
                for M_x, M_y in zip(M_xs, M_ys)
            }

        plt.figure(fig_idx)
        Ms = [M_x * M_y for M_x, M_y in zip(M_xs, M_ys)]
        line_sim = plt.errorbar(
            Ms,
            [
                np.mean(
                    [
                        np.sqrt(M_to_var_sim[idx][M_x, M_y])
                        for idx in sub_car_idxs
                    ]
                )
                for M_x, M_y in zip(M_xs, M_ys)
            ],
            yerr=[
                np.std(
                    [
                        np.sqrt(M_to_var_sim[idx][M_x, M_y])
                        for idx in sub_car_idxs
                    ]
                )
                for M_x, M_y in zip(M_xs, M_ys)
            ],
            color="seagreen",
            ls="-",
            lw=1.5,
            alpha=0.7,
        )
        line_theory = plt.errorbar(
            Ms,
            [
                np.mean(
                    [M_to_var_theory[idx][M_x, M_y] for idx in sub_car_idxs]
                )
                for M_x, M_y in zip(M_xs, M_ys)
            ],
            yerr=[
                np.std([M_to_var_theory[idx][M_x, M_y] for idx in sub_car_idxs])
                for M_x, M_y in zip(M_xs, M_ys)
            ],
            color="r",
            ls="-",
            lw=1.5,
            alpha=0.6,
        )
        line_theory_asy = plt.errorbar(
            [M_x * M_y for M_x, M_y in zip(M_xs, M_ys) if M_x * M_y > 50],
            [
                np.mean(
                    [M_to_var_theory_asy[idx][M_x, M_y] for idx in sub_car_idxs]
                )
                for M_x, M_y in zip(M_xs, M_ys)
                if M_x * M_y > 50
            ],
            yerr=[
                np.std(
                    [M_to_var_theory_asy[idx][M_x, M_y] for idx in sub_car_idxs]
                )
                for M_x, M_y in zip(M_xs, M_ys)
                if M_x * M_y > 50
            ],
            color="blue",
            ls="--",
            lw=1.5,
            alpha=0.6,
        )
    plt.plot(Ms, [1 / np.sqrt(float(M)) for M in Ms], "black", marker=".")
    lines = [
        line_theory_asy,
        line_theory,
        line_sim,
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2,
            linestyle="-",
            marker=".",
            markerfacecolor="black",
            markeredgecolor="black",
        ),
    ]
    labels = [
        "Proposition 2",
        "Proposition 1",
        "Simulation",
        "i.i.d. Rayleigh Fading",
    ]
    plt.legend(lines, labels, loc="upper right")
    plt.xlabel("Array Size, M", fontsize=25)
    plt.ylabel("User Channel Correlation", fontsize=25)
    plt.axis([0, 625, -0.02, 1.0])
    plt.grid(True, alpha=0.3)
    name_pattern = (
        "plots/fig_10_ray_analysis_cor_{array_label}_var_num_paths_{num_paths}"
    )
    plt.savefig(
        name_pattern.format(
            array_label=array_label,
            num_paths=num_paths,
            Delta_x=Delta_x,
            Delta_y=Delta_y,
        ).replace(".", "_")
        + ".pdf"
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fig",
        help="""figures to plot:
                *: plot all figures (Fig. 7-10) in paper Section V.
                fig_7: fig.7 in paper Section V.
                fig_8: fig.8 in paper Section V.
                fig_9: fig.9 in paper Section V.
                fig_10: fig.10 in paper Section V.""",
        choices=["*", "fig_1", "fig_2", "fig_3", "fig_4"],
        default="*",
    )
    parser.add_argument(
        "--num_samples",
        help="number of samples to use, defaults to 20K",
        default=20000,
    )
    parser.add_argument(
        "--num_sample_bars",
        help="number of samples for bar, defaults to 500",
        default=500,
    )
    args = parser.parse_args()

    num_samples = int(args.num_samples)
    num_sample_bars = int(args.num_sample_bars)
    # # plot_2d_los_square_mean(num_samples=num_samples)
    #
    # """"""""""""""""""""""""""
    # " Fig 7 UPA alpha impact "
    # """"""""""""""""""""""""""
    if args.fig == "*" or args.fig == "fig_7":
        alpha_xys = [
            (0.05, 0.05),
            (0.6, 0.6),
            (0.3, 0.3),
            (0.2, 0.2),
            (0.1, 0.1),
        ]
        M_xs_upa = [m for m in range(1, 26, 1)]
        M_ys_upa = [m for m in M_xs_upa]
        plot_2d_square_var(
            array_label="UPA_alpha",
            alpha_xys=alpha_xys,
            fig_idx=1,
            M_xs=M_xs_upa,
            M_ys=M_ys_upa,
            num_paths=1,
            num_samples=num_samples,
            fig_prefix="fig_7",
        )
        plot_2d_square_var(
            array_label="UPA_alpha",
            alpha_xys=alpha_xys,
            fig_idx=2,
            M_xs=M_xs_upa,
            M_ys=M_ys_upa,
            num_paths=3,
            num_samples=num_samples,
            fig_prefix="fig_7",
        )

    # """""""""""""""""""""""""""""""""
    # " Fig 8 ULA alpha impact case   "
    # """""""""""""""""""""""""""""""""
    if args.fig == "*" or args.fig == "fig_8":
        alpha_xys = [
            (0.6, 0.6),
            (0.2, 0.2),
            (0.1, 0.1),
            (0.05, 0.05),
            (0.02, 0.02),
            (0.01, 0.01),
        ]
        M_xs_ula = [m ** 2 for m in range(1, 26, 1)]
        M_ys_ula = [1 for m in M_xs_ula]
        plot_2d_square_var(
            array_label="ULA_alpha",
            alpha_xys=alpha_xys,
            fig_idx=1,
            M_xs=M_xs_ula,
            M_ys=M_ys_ula,
            num_paths=1,
            num_samples=num_samples,
            fig_prefix="fig_8",
        )
        plot_2d_square_var(
            array_label="ULA_alpha",
            alpha_xys=alpha_xys,
            fig_idx=2,
            M_xs=M_xs_ula,
            M_ys=M_ys_ula,
            num_paths=3,
            num_samples=num_samples,
            fig_prefix="fig_8",
        )

    # """""""""""""""""""""""""""""""""
    # " Fig 9 ULA alpha vs D          "
    # """""""""""""""""""""""""""""""""
    if args.fig == "*" or args.fig == "fig_9":
        alpha_xy_and_D_xys = [
            (0.2, 0.2, 0.5, 0.5),
            (0.2, 0.2, 0.25, 0.25),
            (0.1, 0.1, 0.5, 0.5),
            (0.1, 0.1, 0.25, 0.25),
        ]
        M_xs_upa = [m for m in range(1, 26, 1)]
        M_ys_upa = [m for m in M_xs_upa]
        # UPA LOS
        plot_2d_var_D_vs_alpha(
            array_label="alpha_vs_D_UPA_LOS",
            alpha_xy_and_D_xys=alpha_xy_and_D_xys,
            fig_idx=5,
            M_xs=M_xs_upa,
            M_ys=M_ys_upa,
            num_paths=1,
            num_samples=num_samples,
        )
        # ULA NLOS
        M_xs_ula = [m ** 2 for m in range(1, 26, 1)]
        M_ys_ula = [1 for m in M_xs_ula]
        plot_2d_var_D_vs_alpha(
            array_label="alpha_vs_D_ULA_LOS",
            alpha_xy_and_D_xys=alpha_xy_and_D_xys,
            fig_idx=6,
            M_xs=M_xs_ula,
            M_ys=M_ys_ula,
            num_paths=1,
            num_samples=num_samples,
        )

    # """""""""""""""""""""""""""""""""
    # " Fig 10 UPA impact of freq      "
    # """""""""""""""""""""""""""""""""
    if args.fig == "*" or args.fig == "fig_10":
        alpha_xys = [
            (0.05, 0.05),
            (0.6, 0.6),
            (0.3, 0.3),
            (0.15, 0.15),
            (0.1, 0.1),
        ]
        M_xs_upa = [m for m in range(1, 26, 1)]
        M_ys_upa = [m for m in M_xs_upa]
        plot_2d_var_subcarrier_variance(
            array_label="upa_los_bar",
            alpha_xys=alpha_xys,
            fig_idx=7,
            M_xs=M_xs_upa,
            M_ys=M_ys_upa,
            num_paths=1,
            num_samples=num_sample_bars,
        )
        plot_2d_var_subcarrier_variance(
            array_label="upa_nlos_bar_graph",
            alpha_xys=alpha_xys,
            fig_idx=8,
            M_xs=M_xs_upa,
            M_ys=M_ys_upa,
            num_paths=3,
            num_samples=num_sample_bars,
        )
