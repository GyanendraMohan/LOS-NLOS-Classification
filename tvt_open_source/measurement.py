#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measurement code for
```
Shared angles-of-departure in massive MIMO channels:
    Correlation analysis and performance impact
```
@author: Xu Du
"""
# noqa: E501
import numpy as np
import os.path
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.patches import Ellipse
import deepdish as dd
import argparse

from lib import dsp_lib, ota_lib


def plot_heat_map_signed(glb_idx_to_h, M_to_antenna_idx):
    "fig4a: heatmap of inter/intra inter-user channel correlation mean"
    array_size = 64
    global_idx = len(glb_idx_to_h)
    glb_idx0_idx1_to_cor = np.ones([global_idx, global_idx]) * -1
    for glb_idx0 in range(global_idx):
        for glb_idx1 in range(global_idx):
            cor_sampels = [
                0.0
                for _ in range(
                    glb_idx_to_h[glb_idx0].shape[1]
                    * glb_idx_to_h[glb_idx1].shape[1]
                )
            ]
            cur_idx = 0
            for idx0 in range(glb_idx_to_h[glb_idx0].shape[1]):
                for idx1 in range(glb_idx_to_h[glb_idx1].shape[1]):
                    # cha0 = glb_idx_to_h[glb_idx0][:array_size, idx0]
                    # cha1 = glb_idx_to_h[glb_idx1][:array_size, idx1]
                    cha0 = glb_idx_to_h[glb_idx0][
                        M_to_antenna_idx[array_size], idx0
                    ]
                    cha1 = glb_idx_to_h[glb_idx1][
                        M_to_antenna_idx[array_size], idx1
                    ]
                    sample = dsp_lib.ComputeCorrelationSigned(
                        cha0, cha1, normalize=True
                    )
                    cor_sampels[cur_idx] = sample
                    cur_idx += 1
            glb_idx0_idx1_to_cor[glb_idx0][
                glb_idx1
            ] = dsp_lib.get_var_from_sample(cor_sampels)
            print(
                "Between cluster:",
                glb_idx0,
                glb_idx1,
                ", the Correlation level is:",
                glb_idx0_idx1_to_cor[glb_idx0][glb_idx1],
            )
    return glb_idx0_idx1_to_cor


def plot_inter_correlation_signed(
    M_to_antenna_idx,
    M_to_ideal_corr,
    glb_idx_to_label,
    glb_idx_to_h,
    color_list,
    sub_car_idx,
    M_option,
    # If True, plot figs in plots/debug_figs for per sub carrier debugging
    plot_debug=False,
):
    M_idx_to_inter_cluster_correlation = {}
    for glb_idx in range(len(glb_idx_to_label)):
        print(
            "Computing inter cluster correlation for group {}".format(glb_idx)
        )
        for array_size in M_axis:
            cor_sampels = []
            for glb_idx_other in range(len(glb_idx_to_label)):
                if glb_idx == glb_idx_other:
                    continue
                for idx0 in range(glb_idx_to_h[glb_idx].shape[1]):
                    for idx1 in range(glb_idx_to_h[glb_idx_other].shape[1]):
                        # cha0 = glb_idx_to_h[glb_idx][:array_size, idx0]
                        # cha1 = glb_idx_to_h[glb_idx_other][:array_size, idx1]
                        cha0 = glb_idx_to_h[glb_idx][
                            M_to_antenna_idx[array_size], idx0
                        ]
                        if M_option == "square_calib":
                            cha1 = [1.0] * array_size
                        else:
                            cha1 = glb_idx_to_h[glb_idx_other][
                                M_to_antenna_idx[array_size], idx1
                            ]

                        sample = dsp_lib.ComputeCorrelationSigned(
                            cha0, cha1, normalize=True
                        )
                        cor_sampels.append(sample)
            M_idx_to_inter_cluster_correlation[
                array_size, glb_idx
            ] = dsp_lib.get_var_from_sample(cor_sampels)
            print(
                glb_idx_to_label[glb_idx],
                "has ",
                glb_idx_to_h[glb_idx].shape[1],
                "frames and for M=",
                array_size,
                "the Correlation level is:",
                M_idx_to_inter_cluster_correlation[array_size, glb_idx],
            )
    if plot_debug:
        plt.figure(2)
        plt.plot(
            M_axis,
            [M_idx_to_inter_cluster_correlation[M, 0] for M in M_axis],
            color=color_list[0],
            ls="--",
            lw=1.5,
        )
        plt.plot(
            M_axis,
            [M_idx_to_inter_cluster_correlation[M, 1] for M in M_axis],
            color=color_list[1],
            ls="--",
            lw=1.5,
        )
        plt.plot(
            M_axis,
            [M_idx_to_inter_cluster_correlation[M, 2] for M in M_axis],
            color=color_list[2],
            ls="--",
            lw=1.5,
        )
        plt.plot(
            M_axis,
            [M_idx_to_inter_cluster_correlation[M, 3] for M in M_axis],
            color=color_list[3],
            ls="--",
            lw=1.5,
        )
        plt.plot(
            M_axis,
            [M_idx_to_inter_cluster_correlation[M, 4] for M in M_axis],
            color=color_list[4],
            lw=1.5,
        )
        plt.plot(
            M_axis,
            [M_idx_to_inter_cluster_correlation[M, 5] for M in M_axis],
            color=color_list[5],
            lw=1.5,
        )
        plt.plot(
            M_axis,
            [M_idx_to_inter_cluster_correlation[M, 6] for M in M_axis],
            color=color_list[6],
            lw=1.5,
        )
        plt.plot(
            M_axis,
            [M_idx_to_inter_cluster_correlation[M, 7] for M in M_axis],
            color=color_list[7],
            lw=1.5,
        )
        plt.plot(
            M_axis,
            [M_idx_to_inter_cluster_correlation[M, 8] for M in M_axis],
            color=color_list[8],
            lw=1.5,
        )
        "now the raligh fading one"
        plt.plot(
            M_axis, [M_to_ideal_corr[M] for M in M_axis], "black", marker="."
        )
        plt.axis([1, 65, 0, 1])
        plt.grid(True, alpha=0.3)
        plt.xlabel("Array Size, M", fontsize=25)
        plt.ylabel("Inter-Cluster Correlation", fontsize=25)
        plt.savefig(
            "plots/debug_figs/per_sub_inter_cor_" + str(sub_car_idx) + ".pdf"
        )
        plt.close()
    return M_idx_to_inter_cluster_correlation


def plot_intra_correlation_signed(
    M_to_antenna_idx,
    M_to_ideal_corr,
    glb_idx_to_label,
    glb_idx_to_h,
    color_list,
    sub_car_idx,
    # If True, plot figs in plots/debug_figs for per sub carrier debugging
    plot_debug=False,
):
    "Plot 1, plot the average channel between channels in 11"
    M_idx_to_intra_cluster_correlation = {}
    for glb_idx in range(len(glb_idx_to_label)):
        print(
            "Computing intra group correlations for group {}.".format(glb_idx)
        )
        for array_size in M_axis:
            cor_sampels = []
            for idx0 in range(glb_idx_to_h[glb_idx].shape[1]):
                for idx1 in range(idx0 + 1, glb_idx_to_h[glb_idx].shape[1]):
                    sample = dsp_lib.ComputeCorrelationSigned(
                        glb_idx_to_h[glb_idx][
                            M_to_antenna_idx[array_size], idx0
                        ],
                        glb_idx_to_h[glb_idx][
                            M_to_antenna_idx[array_size], idx1
                        ],
                        normalize=True,
                    )
                    cor_sampels.append(sample)
            M_idx_to_intra_cluster_correlation[
                array_size, glb_idx
            ] = dsp_lib.get_var_from_sample(cor_sampels)
            print(
                glb_idx_to_label[glb_idx],
                "has ",
                glb_idx_to_h[glb_idx].shape[1],
                "frames and for M=",
                array_size,
                "the correlation level is:",
                M_idx_to_intra_cluster_correlation[array_size, glb_idx],
            )

    if plot_debug:
        plt.figure(1)
        plt.plot(
            M_axis,
            [M_idx_to_intra_cluster_correlation[M, 0] for M in M_axis],
            color=color_list[0],
            ls="--",
            lw=1.5,
        )
        plt.plot(
            M_axis,
            [M_idx_to_intra_cluster_correlation[M, 1] for M in M_axis],
            color=color_list[1],
            ls="--",
            lw=1.5,
        )
        plt.plot(
            M_axis,
            [M_idx_to_intra_cluster_correlation[M, 2] for M in M_axis],
            color=color_list[2],
            ls="--",
            lw=1.5,
        )
        plt.plot(
            M_axis,
            [M_idx_to_intra_cluster_correlation[M, 3] for M in M_axis],
            color=color_list[3],
            ls="--",
            lw=1.5,
        )
        plt.plot(
            M_axis,
            [M_idx_to_intra_cluster_correlation[M, 4] for M in M_axis],
            color=color_list[4],
            lw=1.5,
        )
        plt.plot(
            M_axis,
            [M_idx_to_intra_cluster_correlation[M, 5] for M in M_axis],
            color=color_list[5],
            lw=1.5,
        )
        plt.plot(
            M_axis,
            [M_idx_to_intra_cluster_correlation[M, 6] for M in M_axis],
            color=color_list[6],
            lw=1.5,
        )
        plt.plot(
            M_axis,
            [M_idx_to_intra_cluster_correlation[M, 7] for M in M_axis],
            color=color_list[7],
            lw=1.5,
        )
        plt.plot(
            M_axis,
            [M_idx_to_intra_cluster_correlation[M, 8] for M in M_axis],
            color=color_list[8],
            lw=1.5,
        )
        "now the Rayleigh fading curve"
        plt.plot(
            M_axis, [M_to_ideal_corr[M] for M in M_axis], "black", marker="."
        )
        plt.axis([1, 65, 0, 1])
        plt.grid(True, alpha=0.3)
        plt.xlabel("Array Size, M", fontsize=25)
        plt.ylabel("Intra-Cluster Correlation", fontsize=25)
        plt.savefig(
            "plots/debug_figs/per_sub_intra_cor_" + str(sub_car_idx) + ".pdf"
        )
        plt.close()
    return M_idx_to_intra_cluster_correlation


def get_ray_fading_stats(sampled_times=3000):
    "Generate the Correlations for realigh fading"
    M_to_ideal_corr = {}
    for array_size in range(1, 65):
        cor_sampels = []
        for i in range(sampled_times):
            cha1 = dsp_lib.rayleigh_cha_gen(1, array_size).flatten()
            cha2 = dsp_lib.rayleigh_cha_gen(1, array_size).flatten()
            cor_sampels.append(
                dsp_lib.ComputeCorrelationSigned(cha1, cha2, normalize=True)
            )
        M_to_ideal_corr[array_size] = dsp_lib.get_var_from_sample(cor_sampels)

    return M_to_ideal_corr


def per_sub_carrier_processing(
    sub_car_idx=50,
    time_sample_rate=100,
    t_sample_offset=0,
    M_axis=np.arange(1, 65),
    M_option="row_based",
):
    "-----------------------locate the csi source file------------------------"
    option_list = [
        "LOS_clusters",
        "NLOS_clusters_part1",
        "NLOS_clusters_part2",
    ]
    global_idx = 0
    glb_idx_to_h = {}
    glb_idx_to_label = {}
    for option in option_list:
        (
            array_config,
            file_path,
            file_name_list,
            label_list,
        ) = ota_lib.OTA_loader(option)
        for file_idx, file_name in enumerate(file_name_list):
            source_ota_file = file_path + file_name
            y = ota_lib.h5pytoCSI_quick(source_ota_file, array_config)
            if time_sample_rate == "auto_for_26_loc":
                sample_num_idx = np.arange(
                    t_sample_offset, y.shape[1], int(y.shape[1] / 26)
                )
            else:
                sample_num_idx = np.arange(
                    t_sample_offset, y.shape[1], time_sample_rate
                )
            print(
                "y shape is:M={}, with frames {} has {} subcarriers".format(
                    y.shape[0], y.shape[1], y.shape[2]
                )
            )
            y = y[:, sample_num_idx, sub_car_idx]
            glb_idx_to_h[global_idx] = y
            glb_idx_to_label[global_idx] = label_list[file_idx]
            global_idx += 1

    M_to_ideal_corr = get_ray_fading_stats()
    M_to_antenna_idx = ota_lib.get_M_to_M_index(M_option)

    "color line gen"
    color_list = cm.rainbow(np.linspace(0, 1, 9))

    M_idx_to_intra_cluster_correlation = plot_intra_correlation_signed(
        M_to_antenna_idx=M_to_antenna_idx,
        M_to_ideal_corr=M_to_ideal_corr,
        glb_idx_to_label=glb_idx_to_label,
        glb_idx_to_h=glb_idx_to_h,
        color_list=color_list,
        sub_car_idx=sub_car_idx,
    )
    M_idx_to_inter_cluster_correlation = plot_inter_correlation_signed(
        M_to_antenna_idx=M_to_antenna_idx,
        M_to_ideal_corr=M_to_ideal_corr,
        glb_idx_to_label=glb_idx_to_label,
        glb_idx_to_h=glb_idx_to_h,
        color_list=color_list,
        sub_car_idx=sub_car_idx,
        M_option=M_option,
    )
    glb_idx0_idx1_to_cor = plot_heat_map_signed(
        glb_idx_to_h=glb_idx_to_h, M_to_antenna_idx=M_to_antenna_idx
    )

    return (
        M_to_ideal_corr,
        M_idx_to_intra_cluster_correlation,
        M_idx_to_inter_cluster_correlation,
        glb_idx0_idx1_to_cor,
    )


def fast_iid_ray(M_axis=np.arange(1, 65), num_sub=52, num_samples=3000):
    output_file_path = "data/"
    file_name = "iid_ray_" + str(num_sub) + "_" + str(num_samples)
    store_name = output_file_path + file_name + ".hdf5"

    if not os.path.isfile(store_name):
        print("###" * 54)
        print("# Cannot find {} in storage. Process now. #".format(store_name))
        print("###" * 54)
        M_to_iid_cor_list = {M: [] for M in M_axis}
        for idx in range(num_sub):
            print("processing suc_car " + str(idx) + " for iid rayleigh")
            M_to_iid_cor = get_ray_fading_stats(sampled_times=num_samples)
            for M in M_axis:
                M_to_iid_cor_list[M].append(np.sqrt(M_to_iid_cor[M]))

        big_dic = {}
        big_dic["iid_res"] = M_to_iid_cor_list
        print("Saving to: " + store_name)
        dd.io.save(store_name, big_dic)

    big_dic = dd.io.load(store_name)
    return big_dic["iid_res"]


def fast_per_sub_carrier_processing(
    sub_car_idx=50,
    time_sample_rate=100,
    t_sample_offset=0,
    M_axis=np.arange(1, 65),
    M_option="row_based",
):
    if M_option == "row_based":
        file_name = "_".join(
            [str(sub_car_idx), str(time_sample_rate), str(t_sample_offset)]
        )
    else:
        file_name = "_".join(
            [
                str(sub_car_idx),
                str(time_sample_rate),
                str(t_sample_offset),
                M_option,
            ]
        )

    store_name = (
        "data/computed_correlation_signed_"
        + "".join(file_name.split("/")[-1:])
        + ".hdf5"
    )
    print("Processed CSI shoule be:", store_name)

    if not os.path.isfile(store_name):
        print("###" * 54)
        print("# Cannot find {} in storage. Process now. #".format(store_name))
        print("###" * 54)
        (
            M_to_ideal_corr,
            M_idx_to_intra_cluster_correlation,
            M_idx_to_inter_cluster_correlation,
            glb_idx0_idx1_to_cor,
        ) = per_sub_carrier_processing(
            sub_car_idx=sub_car_idx,
            time_sample_rate=time_sample_rate,
            t_sample_offset=t_sample_offset,
            M_axis=M_axis,
            M_option=M_option,
        )
        big_dic = {}
        big_dic["ideal"] = M_to_ideal_corr
        big_dic["intra"] = M_idx_to_intra_cluster_correlation
        big_dic["inter"] = M_idx_to_inter_cluster_correlation
        big_dic["heat"] = glb_idx0_idx1_to_cor
        print("Saving mob_csi to: " + store_name)
        dd.io.save(store_name, big_dic)

    big_dic = dd.io.load(store_name)
    M_to_ideal_corr = big_dic["ideal"]
    M_idx_to_intra_cluster_correlation = big_dic["intra"]
    M_idx_to_inter_cluster_correlation = big_dic["inter"]
    glb_idx0_idx1_to_cor = big_dic["heat"]
    return (
        M_to_ideal_corr,
        M_idx_to_intra_cluster_correlation,
        M_idx_to_inter_cluster_correlation,
        glb_idx0_idx1_to_cor,
    )


def plot_correlation(
    sub_index_to_correlation,
    cor_name,
    y_label,
    M_axis,
    color_list=cm.rainbow(np.linspace(0, 1, 9)),
    M_option="row_based",
    show_legend=False,
    plot_iid_error_bar=False,
    fig_name="fig_name",
):
    M_idx_to_cors = {}
    for M in M_axis:
        for cluster_index in range(0, 9):
            M_idx_to_cors[M, cluster_index] = []

    for sample_key in sub_index_to_correlation:
        M_idx_to_cor_per_sample = sub_index_to_correlation[sample_key][cor_name]
        for M, cluster_index in M_idx_to_cor_per_sample:
            M_idx_to_cors[M, cluster_index] += [
                np.sqrt(M_idx_to_cor_per_sample[M, cluster_index])
            ]

    for cluster_idx in range(0, 4):
        plt.errorbar(
            M_axis,
            [np.mean(M_idx_to_cors[M, cluster_idx]) for M in M_axis],
            yerr=[np.std(M_idx_to_cors[M, cluster_idx]) for M in M_axis],
            color=color_list[cluster_idx],
            ls="--",
            lw=1.5,
            label="LOS" + str(cluster_idx + 1),
        )
        # label_curvs.append((label, cur))
    for cluster_idx in range(4, 9):
        plt.errorbar(
            M_axis,
            [np.mean(M_idx_to_cors[M, cluster_idx]) for M in M_axis],
            yerr=[np.std(M_idx_to_cors[M, cluster_idx]) for M in M_axis],
            color=color_list[cluster_idx],
            lw=1.5,
            label="NLOS" + str(cluster_idx + 1 - 4),
        )
        # label_curvs.append(("NLOS" + str(cluster_idx+1), cur))

    "now the raligh fading one"
    if not plot_iid_error_bar:
        plt.plot(
            M_axis,
            [np.sqrt(sub_index_to_correlation[0]["ideal"][M]) for M in M_axis],
            "black",
            marker=".",
        )
    else:
        M_to_iid_cor_list = fast_iid_ray(
            M_axis=M_axis, num_sub=52, num_samples=1000
        )
        plt.errorbar(
            M_axis,
            [np.mean(M_to_iid_cor_list[M]) for M in M_axis],
            yerr=[np.std(M_to_iid_cor_list[M]) for M in M_axis],
            color="black",
            ls="-",
            lw=1.5,
            label="i.i.d. Rayleigh Fading",
        )

    plt.axis([1, 65, 0, 1])
    plt.grid(True, alpha=0.3)
    plt.xlabel("Array Size, M", fontsize=25)
    plt.ylabel(y_label, fontsize=25)
    ax = plt.gca()
    # Plot the legends
    if cor_name == "intra":
        ellipse1 = Ellipse(
            xy=(60, 0.9),
            width=2,
            height=0.18,
            edgecolor="black",
            fc="None",
            lw=1,
        )
        ellipse2 = Ellipse(
            xy=(60, 0.53),
            width=2,
            height=0.2,
            edgecolor="black",
            fc="None",
            lw=1,
        )
        ax.add_patch(ellipse1)
        ax.add_patch(ellipse2)
        "add notes"
        ax.text(52, 0.72, "LOS", color="black", fontsize=14)
        ax.annotate(
            "",
            xy=(60, 0.8),
            xytext=(55, 0.75),
            arrowprops=dict(
                arrowstyle="->",  # linestyle="dashed",
                color="black",
                shrinkA=3,
                shrinkB=3,
                patchA=None,
                patchB=None,
            ),
        )
        ax.text(52, 0.35, "NLOS", color="black", fontsize=14)
        ax.annotate(
            "",
            xy=(60, 0.43),
            xytext=(55, 0.38),
            arrowprops=dict(
                arrowstyle="->",  # linestyle="dashed",
                color="black",
                shrinkA=3,
                shrinkB=3,
                patchA=None,
                patchB=None,
            ),
        )
        ax.text(30, 0.05, "i.i.d. Rayleigh Fading", color="black", fontsize=14)
        ax.annotate(
            "",
            xy=(50, 0.12),
            xytext=(46, 0.1),
            arrowprops=dict(
                arrowstyle="->",  # linestyle="dashed",
                color="black",
                shrinkA=3,
                shrinkB=3,
                patchA=None,
                patchB=None,
            ),
        )
    if show_legend:
        plt.legend(loc="upper right", ncol=2)
    plt.savefig("plots/" + fig_name + ".pdf")
    plt.close()


def plot_heatmap(sub_index_to_correlation, M_option="row_based"):
    index_to_samples = {}

    for sample_key in sub_index_to_correlation:
        glb_idx0_idx1_to_cor = sub_index_to_correlation[sample_key]["heat"]
        for y in range(glb_idx0_idx1_to_cor.shape[0]):
            for x in range(glb_idx0_idx1_to_cor.shape[1]):
                if (x, y) not in index_to_samples:
                    index_to_samples[x, y] = []
                index_to_samples[x, y].append(
                    np.sqrt(glb_idx0_idx1_to_cor[y, x])
                )

    average_map = np.ones([9, 9]) * -1
    std_map = np.ones([9, 9]) * -1
    for y in range(glb_idx0_idx1_to_cor.shape[0]):
        for x in range(glb_idx0_idx1_to_cor.shape[1]):
            average_map[y][x] = np.mean(index_to_samples[x, y])
            std_map[y][x] = np.std(index_to_samples[x, y])

    plt.figure(3)
    cmap_style = "GnBu"
    plt.grid(True, alpha=0.3)
    heatmap = plt.pcolor(average_map, vmin=0, vmax=1, cmap=cmap_style)
    tick_loc = np.arange(0.5, 9.0, 1.0)
    tick_labels = [
        "LOS1",
        "LOS2",
        "LOS3",
        "LOS4",
        "NLOS1",
        "NLOS2",
        "NLOS3",
        "NLOS4",
        "NLOS5",
    ]
    plt.yticks(tick_loc, tick_labels, fontsize=12)
    plt.xticks(tick_loc, tick_labels, fontsize=11)

    for y in range(glb_idx0_idx1_to_cor.shape[0]):
        for x in range(glb_idx0_idx1_to_cor.shape[1]):
            plt.text(
                x + 0.5,
                y + 0.5,
                ("%0.3f" % average_map[y, x]).lstrip("0"),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=16,
            )
    plt.colorbar(heatmap)
    plt.savefig("plots/fig_4a_cor_mean_heatmap.pdf")
    plt.close()

    plt.figure(3)
    plt.grid(True, alpha=0.3)
    heatmap = plt.pcolor(std_map, vmin=0, vmax=1, cmap=cmap_style)
    tick_loc = np.arange(0.5, 9.0, 1.0)
    tick_labels = [
        "LOS1",
        "LOS2",
        "LOS3",
        "LOS4",
        "NLOS1",
        "NLOS2",
        "NLOS3",
        "NLOS4",
        "NLOS5",
    ]
    plt.yticks(tick_loc, tick_labels, fontsize=12)
    plt.xticks(tick_loc, tick_labels, fontsize=11)

    for y in range(std_map.shape[0]):
        for x in range(std_map.shape[1]):
            plt.text(
                x + 0.5,
                y + 0.5,
                ("%0.3f" % std_map[y, x]).lstrip("0"),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=16,
            )
    plt.colorbar(heatmap)
    plt.savefig("plots/fig_4b_cor_std_heatmap.pdf")
    plt.close()


def channel_preprocessing(time_sample_rate, M_axis, array_option):
    # 52 data subcarrier index
    sub_car_idxs = range(52)
    sub_car_idxs = np.random.permutation(sub_car_idxs)
    sub_index_to_correlation = {}
    t_sample_offsets = [0]

    for sub_car_idx in sub_car_idxs:
        for t_sample_offset in t_sample_offsets:
            (
                M_to_ideal_corr,
                M_idx_to_intra_cluster_correlation,
                M_idx_to_inter_cluster_correlation,
                glb_idx0_idx1_to_cor,
            ) = fast_per_sub_carrier_processing(
                sub_car_idx=sub_car_idx,
                time_sample_rate=time_sample_rate,
                t_sample_offset=t_sample_offset,
                M_axis=M_axis,
                M_option=array_option,
            )
            sub_index_to_correlation[sub_car_idx] = {}
            sub_index_to_correlation[sub_car_idx][
                "intra"
            ] = M_idx_to_intra_cluster_correlation
            sub_index_to_correlation[sub_car_idx][
                "inter"
            ] = M_idx_to_inter_cluster_correlation
            sub_index_to_correlation[sub_car_idx]["heat"] = glb_idx0_idx1_to_cor
            sub_index_to_correlation[sub_car_idx]["ideal"] = M_to_ideal_corr

    return sub_index_to_correlation


def get_array_indexs(array_option):
    if array_option == "square" or array_option == "square_calib":
        M_axis = [1, 4, 9, 16, 25, 36, 49, 64]
    else:
        M_axis = range(1, 65)

    return M_axis


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fig",
        help="""Figures to plot.
                        *: plot both Fig.3 and Fig.4.
                        fig_3: plot Fig.3 in Section III-B.
                        fig_4: plot Fig.4 in Section III-B.
                        """,
        choices=["*", "fig_3", "fig_4"],
        default="*",
    )
    parser.add_argument(
        "--array_option",
        help="Array downsampling method. Defaults to square.",
        choices=["square", "square_calib", "row_based", "col_based"],
        default="square",
    )
    args = parser.parse_args()

    time_sample_rate = "auto_for_26_loc"
    M_axis = get_array_indexs(args.array_option)
    pre_compute_res = channel_preprocessing(
        time_sample_rate, M_axis, args.array_option
    )

    if args.fig == "*" or args.fig == "fig_3":
        plot_correlation(
            pre_compute_res,
            "intra",
            "Intra-Cluster Correlation",
            M_axis,
            M_option=args.array_option + str(time_sample_rate),
            show_legend=False,
            plot_iid_error_bar=True,
            fig_name="fig_3a_ota_intra_channel_cor_vs_M",
        )
        plot_correlation(
            pre_compute_res,
            "inter",
            "Inter-Cluster Correlation",
            M_axis,
            M_option=args.array_option + str(time_sample_rate),
            show_legend=True,
            plot_iid_error_bar=True,
            fig_name="fig_3b_ota_inter_channel_cor_vs_M",
        )

    if args.fig == "*" or args.fig == "fig_4":
        # figure 4
        plot_heatmap(pre_compute_res, M_option=args.array_option)
