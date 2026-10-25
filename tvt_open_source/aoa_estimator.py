#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use MUISC algo to get AoA estimation based on measured channel,
which is Fig.5 in
```
Shared angles-of-departure in massive MIMO channels:
    Correlation analysis and performance impact
```
@author: Xu Du
"""
import numpy as np
from lib import dsp_lib, ota_lib
import argparse
import logging

EST_NUM_SIG = 4
THETA_GRID_NUM = 181  # grid in elevation domain
BETA_GRID_NUM = 181  # grid in azimuth domain
SUC_CARRIER_INDEX = 50  # sub carrier idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster_name",
        help="""Cluster Name. Must be one of
        LOS 1, LOS 2, LOS 3, LOS 4, NLOS 1, NLOS 2, NLOS 3, NLOS4, NLOS 5""",
        choices=[
            "LOS 1",
            "LOS 2",
            "LOS 3",
            "LOS 4",
            "NLOS 1",
            "NLOS 2",
            "NLOS 3",
            "NLOS 4",
            "NLOS 5",
        ],
        required=True,
    )
    parser.add_argument(
        "--location_idx",
        help="""Location index Name.
        User location index inside a cluster, which can be int from 0-26.""",
        choices=[str(i) for i in range(27)],
        required=True,
    )
    args = parser.parse_args()

    all_cluster_channels = ota_lib.get_all_cluster_measured_channel()
    [x_loc, y_loc, M, freq_in, _, _, ant_idx] = ota_lib.array_config_gen(
        0,
        0,
        array_option="TVT_LOS_array"
        if args.cluster_name.startswith("LOS")
        else "TVT_NLOS_array",
        remove_row=True,
    )

    if args.cluster_name in all_cluster_channels:
        # all locations in a single cluster
        single_cluster_channels = all_cluster_channels[args.cluster_name]
    else:
        raise ValueError(
            "Failed to find measured channel in cluster " + args.cluster_name
        )

    if int(args.location_idx) in single_cluster_channels:
        # get channels of a single user
        single_location_channel = single_cluster_channels[
            int(args.location_idx)
        ]
    else:
        raise ValueError(
            "Location index not found in cluster " + args.location_idx
        )

    # sample a frame of one subcarrier for plotting
    # we will use the first frame for each location and SUC_CARRIER_INDEX-th
    # subcarrier
    frame_to_estimation = single_location_channel[
        :, 0, SUC_CARRIER_INDEX
    ].reshape([64, 1])

    logging.info(
        "processing cluster :",
        args.cluster_name,
        "location:",
        args.location_idx,
    )

    Pmusic, aoa_loc = dsp_lib.music_est(
        frame_to_estimation,
        EST_NUM_SIG,
        M,
        freq_in,
        THETA_GRID_NUM,
        BETA_GRID_NUM,
        x_loc,
        y_loc,
        peak_aoa_refine=False,
    )
    v_range = np.array([-3, 0])
    fig_name = "plots/fig5s/ota_aoa_music_"
    fig_name += args.cluster_name + "_" + args.location_idx + ".pdf"

    dsp_lib.my_pcolor(
        Pmusic,
        THETA_GRID_NUM,
        BETA_GRID_NUM,
        v_range=v_range,
        fig_name=fig_name,
    )
