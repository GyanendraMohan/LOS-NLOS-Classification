import numpy as np
from numpy import linalg as LA
import sys
import matplotlib.pyplot as plt
import scipy.ndimage.filters as img_filters


def rayleigh_cha_gen(K, M):
    h = (
        np.random.normal(0.0, np.sqrt(0.5), (K, M))
        + np.random.normal(0.0, np.sqrt(0.5), (K, M)) * 1j
    )
    return h


def get_random_phis_coeff():
    phi = np.random.uniform(0, np.pi * 2)
    return np.exp(1.0j * phi)


def ComputeCorrelationSigned(h1, h2, normalize):
    if len(h1) != len(h2):
        sys.exit("Unmatched vector length!")
    if normalize:
        h1 = h1 / LA.norm(h1)
        h2 = h2 / LA.norm(h2)
    return get_random_phis_coeff() * np.dot(np.conj(h1), h2)


def random_sin_theta(alpha):
    sin_theta0 = np.random.uniform(-1, 1, 1)[0]
    sin_theta1 = sin_theta0 + np.random.uniform(-alpha, alpha, 1)[0]
    return sin_theta0, sin_theta1


def compute_vector_correlation(h1, h2):
    if len(h1) != len(h2):
        sys.exit("Unmatched vector length!")
    return np.dot(np.conj(h1), h2) / float(len(h1))


def sinc(x):
    if not x:
        return 1
    else:
        return np.sin(x) / x


def get_eta(alpha, Delta, M):
    eta = 0.0
    for m1 in range(M):
        for m2 in range(M):
            eta += sinc((m1 - m2) * alpha * Delta * 2 * np.pi) / M ** 2
    return eta


def theory_res(M_x, alpha_x, Delta_x, M_y, alpha_y, Delta_y, num_paths):
    var = 1.0 / float(num_paths) * get_eta(alpha_x, Delta_x, M_x) * get_eta(
        alpha_y, Delta_y, M_y
    ) + (num_paths - 1) / float(num_paths) * get_eta(1, Delta_x, M_x) * get_eta(
        1, Delta_y, M_y
    )
    return 0, var


def get_eta_asy(alpha, delta, M):
    return 1.0 / (M * delta * 2 * alpha) if M > 1 else 1.0


def theory_res_asy(M_x, alpha_x, Delta_x, M_y, alpha_y, Delta_y, num_paths):
    var = 1.0 / num_paths * get_eta_asy(alpha_x, Delta_x, M_x) * get_eta_asy(
        alpha_y, Delta_y, M_y
    ) + (num_paths - 1) / float(num_paths) * get_eta_asy(
        1, Delta_x, M_x
    ) * get_eta_asy(
        1, Delta_y, M_y
    )
    return 0, var


def get_var_from_sample(samples):
    return np.var(samples).astype(np.float64)


def scanning_grid(theta_grid_num, beta_grid_num):
    theta_grid = (
        np.arange(0, theta_grid_num) * np.pi / (theta_grid_num - 1)
    ) - np.pi / 2
    beta_grid = (
        np.arange(0, beta_grid_num) * np.pi / (beta_grid_num - 1)
    ) - np.pi / 2
    return theta_grid, beta_grid


def my_pcolor(
    Pmusic,
    theta_grid_num,
    beta_grid_num,
    v_range=np.array([-3, 0]),
    peak_loc=np.zeros([0, 2], dtype=int),
    saving_pdf=False,
    fig_name="test_aoa_plot",
):
    "change to log scale"
    Pmusic = 10 * np.log10(Pmusic / np.max(Pmusic))

    v_min_val, v_max_val = v_range
    " generating plot grid "
    [theta_grid, beta_grid] = scanning_grid(theta_grid_num, beta_grid_num)
    theta_grid = theta_grid / np.pi * 180
    beta_grid = beta_grid / np.pi * 180
    [theta_mesh, beta_mesh] = np.meshgrid(theta_grid, beta_grid)
    "plotting"
    fig_10 = plt.figure(10)
    plt.clf()
    pcol = plt.pcolormesh(
        theta_grid,
        beta_grid,
        Pmusic,
        vmin=v_min_val,
        vmax=v_max_val,
        cmap="RdBu_r",
    )
    plt.axis([-90, 90, -90, 90])
    plt.yticks([-90, -45, 0, 45, 90], ["-90°", "-45°", "0°", "45°", "90°"])
    plt.xticks([-90, -45, 0, 45, 90], ["-90°", "-45°", "0°", "45°", "90°"])
    plt.ylabel("Elevation Angle", fontsize=25)
    plt.xlabel("Azimuth Angle", fontsize=25)
    pcol.set_edgecolor("face")
    plt.colorbar()
    if peak_loc.shape[0] > 0.1:
        for plt_aoa_idx in range(peak_loc.shape[0]):
            plt.plot(peak_loc[plt_aoa_idx][0], peak_loc[plt_aoa_idx][1], "ro")

    plt.savefig(fig_name)
    plt.close(fig_10)


def music_est(
    y,
    est_num_sig,
    M,
    freq_in,
    theta_grid_num,
    beta_grid_num,
    x_loc,
    y_loc,
    peak_aoa_refine=True,
):
    [theta_grid, beta_grid] = scanning_grid(theta_grid_num, beta_grid_num)
    std_lambda = 299792458 / freq_in[0]

    R = np.dot(y, y.conj().T)
    J = np.fliplr(np.eye(M))
    R = R + np.dot(np.dot(J, np.conjugate(R)), J)
    # np.dot(a,V)-np.dot(V,np.diag(W)) should be closed to 0s
    [W, V] = np.linalg.eig(R)
    # double check to make sure the eig value descends
    "Python sort will sort based on the ascending array"
    # W_new  = np.sort(abs(W),axis=0);
    W_indx = np.argsort(abs(W), axis=0)
    #  print(abs(W[W_indx[-8:]]));
    V = V[:, W_indx]
    VV = V[:, 0 : (M - est_num_sig)]

    Pmusic = np.zeros([theta_grid_num, beta_grid_num])
    for theta_idx, theta in enumerate(theta_grid):  # x should be colom idx
        for beta_idx, beta in enumerate(beta_grid):
            SS_x_sin_mtx = np.sin(theta) * np.ones([1, M])
            SS_y_sin_mtx = np.sin(beta) * np.ones([1, M])

            SS_x = (x_loc * SS_x_sin_mtx) * 2 * np.pi / std_lambda
            SS_y = (y_loc * SS_y_sin_mtx) * 2 * np.pi / std_lambda
            SS = np.exp(-1j * (SS_x + SS_y))

            PP = np.dot(np.dot(SS, VV), np.dot(VV.conj().T, SS.conj().T))
            # print(PP)
            Pmusic[beta_idx, theta_idx] = abs(1 / PP)

    if peak_aoa_refine:
        "refine the resolution for the best aoa"
        peak_loc = my_local_max_finder(Pmusic)
        for aoa_idx, refined_aoa_loc in enumerate(peak_loc):
            theta_list = np.arange(
                refined_aoa_loc[0] - 2, refined_aoa_loc[0] + 2, 0.01
            )
            beta_list = np.arange(
                refined_aoa_loc[1] - 2, refined_aoa_loc[1] + 2, 0.01
            )
            peak_val = 0
            for theta in theta_list:
                for beta in beta_list:
                    SS_x_sin_mtx = np.sin(theta / 180.0 * np.pi) * np.ones(
                        [1, M]
                    )
                    SS_y_sin_mtx = np.sin(beta / 180.0 * np.pi) * np.ones(
                        [1, M]
                    )
                    SS_x = (x_loc * SS_x_sin_mtx) * 2 * np.pi / std_lambda
                    SS_y = (y_loc * SS_y_sin_mtx) * 2 * np.pi / std_lambda
                    SS = np.exp(-1j * (SS_x + SS_y))
                    PP = np.dot(
                        np.dot(SS, VV), np.dot(VV.conj().T, SS.conj().T)
                    )
                    if peak_val < abs(1 / PP):
                        peak_val = abs(1 / PP)
                        refined_aoa_loc[0] = theta
                        refined_aoa_loc[1] = beta
            peak_loc[aoa_idx] = refined_aoa_loc
    else:
        peak_loc = "NONE"
    return Pmusic, peak_loc


def my_local_max_finder(
    Pmusic, theta_grid_num=181, beta_grid_num=181, select_ratio=-10
):
    "Check Pmusic is"
    if np.min(Pmusic) < 0:
        sys.exit("Pmusic cannot be negative!")
    "Normlized Pmusic if needed"
    if np.max(Pmusic) != 1:
        Pmusic = Pmusic / np.max(abs(Pmusic))

        power_thrshold = 0.5
        neighborhood_size = 5
        max_p = img_filters.maximum_filter(Pmusic, neighborhood_size)
        p_mask = Pmusic == max_p
        peak_val = Pmusic[p_mask == True]
        peak_val = np.sort(peak_val)
        peak_val = peak_val[::-1]
        "change to descending order"

        "generate beta theta map"
        [theta_grid, beta_grid] = scanning_grid(theta_grid_num, beta_grid_num)
        theta_grid = theta_grid / np.pi * 180
        beta_grid = beta_grid / np.pi * 180

        peak_loc_idx = np.zeros([0, 2], dtype=int)
        "store the aoa lcoations"

        for peak_val_pt in peak_val:
            if peak_val_pt > power_thrshold:
                peak_loc_pt = np.where(Pmusic == peak_val_pt)
                peak_loc_pt = np.array(
                    [int(peak_loc_pt[1]), int(peak_loc_pt[0])]
                )
                peak_loc_idx = np.vstack((peak_loc_idx, peak_loc_pt))

        "tranlate idx to beta, theta"
        peak_loc = np.zeros(peak_loc_idx.shape, dtype=float)
        for peak_idx in range(peak_loc.shape[0]):
            peak_loc[peak_idx][0] = theta_grid[peak_loc_idx[peak_idx][0]]
            peak_loc[peak_idx][1] = beta_grid[peak_loc_idx[peak_idx][1]]
    return peak_loc
