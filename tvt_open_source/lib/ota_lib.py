import numpy as np
import h5py
import sys
import time
from .argos_lib import samps2csi_large
import os


def array_config_gen(M_x, M_y, array_option="square", remove_row=False):
    "find the x,y locations of antenna elements"
    M = M_x * M_y
    if array_option == "square":
        if M_x == 1 and M_y == 1:
            return [0], [0]
        ant_sp_x = 0.0635
        ant_sp_y = 0.0635
        # location mtx for x and y direction
        x_loc = ant_sp_x * np.array(
            [
                np.arange(0, M_x),
            ]
            * M_y
        )
        y_loc = (
            ant_sp_y
            * np.array(
                [
                    np.arange(0, M_y),
                ]
                * M_x
            ).transpose()
        )

        x_loc = np.squeeze(np.reshape(x_loc.transpose(), [1, M]))
        y_loc = np.squeeze(np.reshape(y_loc.transpose(), [1, M]))
        return x_loc, y_loc
    elif array_option == "old_NASA_array":
        ant_sp_x = 0.0635
        ant_sp_y = 0.0635
        # location mtx for x and y direction
        x_loc = ant_sp_x * np.array(
            [
                np.arange(0, M_x),
            ]
            * M_y
        )
        y_loc = (
            ant_sp_y
            * np.array(
                [
                    np.arange(0, M_y),
                ]
                * M_x
            ).transpose()
        )

        x_loc = np.squeeze(np.reshape(x_loc, [1, M]))
        y_loc = np.squeeze(np.reshape(y_loc, [1, M]))
    elif array_option == "TVT_LOS_array" or array_option == "TVT_NLOS_array":
        ant_sp_x = 0.0635
        ant_sp_y = -0.0635
        M_x = 8
        M_y = 9
        M = M_x * M_y
        # location mtx for x and y direction
        x_loc = ant_sp_x * np.array(
            [
                np.arange(0, M_x),
            ]
            * M_y
        )
        y_loc = (
            ant_sp_y
            * np.array(
                [
                    np.arange(0, M_y),
                ]
                * M_x
            ).transpose()
        )

        x_loc = np.squeeze(np.reshape(x_loc, [1, M]))
        y_loc = np.squeeze(np.reshape(y_loc, [1, M]))

        if remove_row:
            M = M - 8
            x_loc = np.delete(x_loc, [32, 33, 34, 35, 36, 37, 38, 39])
            y_loc = np.delete(y_loc, [32, 33, 34, 35, 36, 37, 38, 39])
            ant_idx = np.arange(4, 72)
            ant_idx = np.delete(ant_idx, [32, 33, 34, 35])
            "remove the whole row"
        else:
            "remove config of Board W3-a-00441 which failed in the exp"
            M = M - 4
            x_loc = np.delete(x_loc, [32, 33, 34, 35])
            y_loc = np.delete(y_loc, [32, 33, 34, 35])
            ant_idx = np.arange(4, 72)

        freq_in = np.array([2.4835 * (10 ** 9)])
        if array_option == "TVT_LOS_array":
            best_mea_idx = 0
            best_cal_idx = 3
        elif array_option == "TVT_NLOS_array":
            best_mea_idx = 3
            best_cal_idx = 0
            "FOR NLOS, ADDED 20DB+15DB ATTEN"
        return x_loc, y_loc, M, freq_in, best_cal_idx, best_mea_idx, ant_idx
    else:
        sys.exit("Unknown Array Config!")


def OTA_loader(options):
    print("beign to process file_option")
    file_path = "raw_traces/"
    if options == "LOS_clusters":
        file_name_list = [
            "channe14_LOS_cluster1.hdf5",  # LOS 1,
            "channe14_LOS_cluster2.hdf5",  # LOS 2,
            "channe14_LOS_cluster3.hdf5",  # LOS 3,
            "channe14_LOS_cluster4.hdf5",  # LOS 4,
        ]
        label_list = ["LOS 1", "LOS 2", "LOS 3", "LOS 4"]
        array_config = "TVT_LOS_array"
    elif options == "NLOS_clusters_part1":
        file_name_list = ["channe14_NLOS_cluster1.hdf5"]
        array_config = "TVT_NLOS_array"
        label_list = ["NLOS 1"]
    elif options == "NLOS_clusters_part2":
        file_name_list = [
            "channe14_NLOS_cluster2.hdf5",
            "channe14_NLOS_cluster3.hdf5",
            "channe14_NLOS_cluster4.hdf5",
            "channe14_NLOS_cluster5.hdf5",
        ]
        array_config = "TVT_NLOS_array"
        label_list = ["NLOS 2", "NLOS 3", "NLOS 4", "NLOS 5"]
    elif options == "AoAs_used":
        file_name_list = [
            "channe14_NLOS_cluster2.hdf5",
            "channe14_NLOS_cluster3.hdf5",
            "channe14_NLOS_cluster4.hdf5",
            "channe14_NLOS_cluster5.hdf5",
        ]
        array_config = "TVT_NLOS_array"
        label_list = ["NLOS 2", "NLOS 3", "NLOS 4", "NLOS 5"]

    return array_config, file_path, file_name_list, label_list


def h5pytoCSI(file_name, array_config, CSI_method="mob", remove_row=True):
    # load the dara from the h5py file
    start_time = time.time()

    print("Tesing " + file_name + " now.")
    h5log = h5py.File(file_name, "r")

    samps_per_user = h5log.attrs["samples_per_user"]
    num_users = h5log.attrs["num_mob_ant"]
    noise_meas_en = h5log.attrs.get("measured_noise", 1)

    # Compute CSI for each user and get a nice numpy array
    csi, iq = samps2csi_large(
        h5log["Pilot_Samples"], num_users + noise_meas_en, samps_per_user
    )
    # Returns csi with Frame, User, LTS (there are 2), BS ant, Subcarrier
    # also, iq samples nicely chunked out, same dims, but subcarrier is sample.
    # noise = csi[:, -1, :, :, :]	 # noise is last set of data.
    if CSI_method == "average":
        userCSI = np.mean(csi[:, :num_users, :, :, :], 2)
        # don't include noise, average over both LTSs
    elif CSI_method == "mob":
        userCSI = csi[:, :num_users, 0, :, :]
        # userCSI_test = csi[:, :num_users, 1, :, :]
    else:
        sys.exit("Unknown CSI obtaining methoded!")

    [
        x_loc,
        y_loc,
        M,
        freq_in,
        best_cal_idx,
        best_mea_idx,
        ant_idx,
    ] = array_config_gen(0, 0, array_option=array_config, remove_row=remove_row)

    print("Loading CSI takes", time.time() - start_time, "s.")
    print("userCSI size:", userCSI.shape)
    # print('userCSI_test size:', userCSI_test.shape)

    mob_csi = extract_mob_CSI(userCSI, best_cal_idx, best_mea_idx, ant_idx, M)

    return mob_csi


def extract_mob_CSI(
    userCSI,
    best_cal_idx,
    best_mea_idx,
    ant_idx,
    M,
    sub_idx_list=np.arange(0, 52),
    wind_lower=200,
    calibrateion=True,
):
    "-------------------Extra the obtained CSI---------------"
    "userCSI is of shape frm X mob_ant X bs_ant X sub-car"
    start_time = time.time()

    # Normalzie with claibration node
    [max_frm, user_num, bs_ant_num, sub_car_num] = userCSI.shape

    "Discard the first few which might still AGC setting period"
    wind_upper = max_frm - 1
    "Remove the last frame where all measureed is 0"
    # clibrate the antenna
    flip_ant_on_borad = False
    print("The following sub-cars are selected:", sub_idx_list)
    y_out = np.empty(
        [
            M,
            int(np.floor(wind_upper - wind_lower)),
            int(sub_idx_list.shape[0]),
        ],
        dtype=complex,
    )

    abs_frm_idx = 0
    for frm_idx in range(wind_lower, wind_upper):
        for sub_idx in sub_idx_list:
            y_mea = np.fliplr(userCSI[frm_idx, :, :, sub_idx]).T
            y_pro = y_mea[ant_idx, :]

            if flip_ant_on_borad:
                print("\n Antenna idx on each board is fliped! \n")
                for y_index in range(22):
                    y_pro[y_index * 4, :] = y_mea[y_index * 4 + 3, :]
                    y_pro[y_index * 4 + 1, :] = y_mea[y_index * 4 + 2, :]
                    y_pro[y_index * 4 + 2, :] = y_mea[y_index * 4 + 1, :]
                    y_pro[y_index * 4 + 3, :] = y_mea[y_index * 4, :]

            y_out[:, abs_frm_idx, sub_idx] = y_pro[:, best_mea_idx]
            "if calibration, remove phase offsets between antenna arrays"
            if calibrateion:
                y_cal = y_pro[:, best_cal_idx] / abs(y_pro[:, best_cal_idx])
                y_out[:, abs_frm_idx, sub_idx] = (
                    y_out[:, abs_frm_idx, sub_idx] / y_cal
                )

        abs_frm_idx += 1
    print(
        "CSI calibration takes",
        1000 * (time.time() - start_time),
        " ms for frm",
        wind_lower,
        " to frame ",
        wind_upper,
    )
    return y_out


def h5pytoCSI_quick(file_name, array_config):
    output_file_path = "channel_data/processed_"
    store_name = file_name[file_name.find("/") + 1 :]
    store_name = store_name[: store_name.rfind(".")]
    store_name = output_file_path + store_name + ".hdf5"
    print("Processed CSI shoule be:", store_name)

    if not os.path.isfile(store_name):
        mob_csi = h5pytoCSI(file_name, array_config, CSI_method="mob")
        print("Saving mob_csi to: " + store_name)
        csi_file = h5py.File(store_name, "w")
        csi_file.create_dataset("mob_csi", data=mob_csi)
        csi_file.close()

    csi_file = h5py.File(store_name, "r")
    mob_csi = csi_file["mob_csi"][:]
    csi_file.close()
    return mob_csi


def get_M_to_M_index(M_option):
    M_to_M_index = {}
    if M_option == "row_based":
        for M in range(1, 65):
            M_to_M_index[M] = [i for i in range(M)]
    elif M_option == "col_based":
        for M in range(1, 65):
            num_col = int(M / 8)
            M_to_M_index[M] = []
            for col_idx in range(num_col):
                M_to_M_index[M] += [
                    col_idx + 8 * row_idx for row_idx in range(8)
                ]

            reminder = M % 8
            for row_idx in range(reminder):
                M_to_M_index[M] += [row_idx * 8 + num_col]

            M_to_M_index[M] = sorted(M_to_M_index[M])

            if len(M_to_M_index[M]) != M:
                sys.exit("Error in get mapping!")

    elif M_option == "square" or M_option == "square_calib":
        M_to_M_index = {
            1: [0],
            4: [0, 1, 8, 9],
            9: [0, 1, 2, 8, 9, 10, 16, 17, 18],
            16: [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27],
            25: [
                0,
                1,
                2,
                3,
                4,
                8,
                9,
                10,
                11,
                12,
                16,
                17,
                18,
                19,
                20,
                24,
                25,
                26,
                27,
                28,
                32,
                33,
                34,
                35,
                36,
            ],
            36: [
                0,
                1,
                2,
                3,
                4,
                5,
                8,
                9,
                10,
                11,
                12,
                13,
                16,
                17,
                18,
                19,
                20,
                21,
                24,
                25,
                26,
                27,
                28,
                29,
                32,
                33,
                34,
                35,
                36,
                37,
                40,
                41,
                42,
                43,
                44,
                45,
            ],
            49: [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
            ],
            64: [i for i in range(64)],
        }
    else:
        sys.exit("Unknown M_option of" + str(M_option))

    return M_to_M_index


def get_single_cluster_channel_mobile():
    """
    Get a channel of a user who are randomly moving in each cluster
    """
    option_list = [
        "LOS_clusters",
        "NLOS_clusters_part1",
        "NLOS_clusters_part2",
    ]
    "load the estimated channel y"
    cluster_name_to_h = {}
    for option in option_list:
        (
            array_config,
            file_path,
            file_name_list,
            label_list,
        ) = OTA_loader(option)
        for file_idx, file_name in enumerate(file_name_list):
            source_ota_file = file_path + file_name
            y = h5pytoCSI_quick(source_ota_file, array_config)
            cluster_name_to_h[label_list[file_idx]] = y
    return cluster_name_to_h


def get_all_cluster_measured_channel():
    option_list = [
        "LOS_clusters",
        "NLOS_clusters_part1",
        "NLOS_clusters_part2",
    ]
    "load the estimated channel y"
    cluster_name_to_h = {}
    for option in option_list:
        (
            array_config,
            file_path,
            file_name_list,
            label_list,
        ) = OTA_loader(option)
        for file_idx, file_name in enumerate(file_name_list):
            source_ota_file = file_path + file_name
            y = h5pytoCSI_quick(source_ota_file, array_config)

            per_location_channel = {}
            for location_idx in range(26):
                per_location_begin_frame = int(y.shape[1] / 26) * location_idx
                per_location_end_frame = int(y.shape[1] / 26) * (
                    location_idx + 1
                )
                per_location_channel[location_idx] = y[
                    :, per_location_begin_frame:per_location_end_frame, :
                ]

            cluster_name_to_h[label_list[file_idx]] = per_location_channel
    return cluster_name_to_h
