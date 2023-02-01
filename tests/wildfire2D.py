import time

import numpy as np
from scipy.special import eval_hermite
import os
import sys
from random import randint
from Helper import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

impath = "../plots/images_wildfire2D/"
os.makedirs(impath, exist_ok=True)

data_path = os.path.abspath(".") + '/wildfire_data/2D/'

cmap = 'YlOrRd'
# cmap = 'YlGn'


class wildfire2D:
    def __init__(self, q_test, shifts_test, param_test_val, var):
        dat1_train = np.load(data_path + 'SnapShotMatrix540.npy')
        dat2_train = np.load(data_path + 'SnapShotMatrix550.npy')
        dat3_train = np.load(data_path + 'SnapShotMatrix560.npy')
        dat4_train = np.load(data_path + 'SnapShotMatrix570.npy')
        dat5_train = np.load(data_path + 'SnapShotMatrix580.npy')
        self.grid_1D = np.load(data_path + '1D_Grid.npy', allow_pickle=True)
        self.grid_2D = np.load(data_path + '2D_Grid.npy', allow_pickle=True)
        self.x = self.grid_1D[0]
        self.y = self.grid_1D[1]
        self.X = self.grid_2D[0]
        self.Y = self.grid_2D[1]
        self.t = np.load(data_path + 'Time.npy')
        delta1_train = np.load(data_path + 'Shifts540.npy')
        delta2_train = np.load(data_path + 'Shifts550.npy')
        delta3_train = np.load(data_path + 'Shifts560.npy')
        delta4_train = np.load(data_path + 'Shifts570.npy')
        delta5_train = np.load(data_path + 'Shifts580.npy')

        self.var = var
        self.Nx = np.size(self.x)
        self.Ny = np.size(self.y)
        self.Nt = np.size(self.t)
        self.x_c = self.x[-1] // 2
        self.y_c = self.y[-1] // 2

        # Test data
        self.param_test_val = param_test_val
        self.mu_vecs_test = np.asarray([self.param_test_val])
        self.q_test = np.reshape(np.transpose(q_test), newshape=[self.Nt, 2, self.Nx, self.Ny], order="F")
        self.q_test = np.transpose(np.reshape(np.squeeze(self.q_test[:, var, :, :]), newshape=[self.Nt, -1], order="F"))
        self.shifts_test = shifts_test
        self.q_polar_test = None

        # Train data
        self.Nsamples_train = 5
        self.NumFrames = 2
        self.mu_vecs_train = np.asarray([540, 550, 560, 570, 580])
        self.params_train = [np.squeeze(np.asarray([[np.ones_like(self.t) * mu], [self.t]])) for mu in
                             self.mu_vecs_train]
        self.params_train = np.concatenate(self.params_train, axis=1)
        self.shifts_train = np.zeros((self.NumFrames, 2, self.Nsamples_train * self.Nt), dtype=float)
        self.shifts_train[0] = np.concatenate((delta1_train[0], delta2_train[0], delta3_train[0],
                                               delta4_train[0], delta5_train[0]), axis=1)
        self.shifts_train[1] = np.concatenate((delta1_train[1], delta2_train[1], delta3_train[1],
                                               delta4_train[1], delta5_train[1]), axis=1)
        dat1_train = np.reshape(np.transpose(dat1_train), newshape=[self.Nt, 2, self.Nx, self.Ny], order="F")
        dat1_train = np.transpose(np.reshape(np.squeeze(dat1_train[:, var, :, :]), newshape=[self.Nt, -1], order="F"))
        dat2_train = np.reshape(np.transpose(dat2_train), newshape=[self.Nt, 2, self.Nx, self.Ny], order="F")
        dat2_train = np.transpose(np.reshape(np.squeeze(dat2_train[:, var, :, :]), newshape=[self.Nt, -1], order="F"))
        dat3_train = np.reshape(np.transpose(dat3_train), newshape=[self.Nt, 2, self.Nx, self.Ny], order="F")
        dat3_train = np.transpose(np.reshape(np.squeeze(dat3_train[:, var, :, :]), newshape=[self.Nt, -1], order="F"))
        dat4_train = np.reshape(np.transpose(dat4_train), newshape=[self.Nt, 2, self.Nx, self.Ny], order="F")
        dat4_train = np.transpose(np.reshape(np.squeeze(dat4_train[:, var, :, :]), newshape=[self.Nt, -1], order="F"))
        dat5_train = np.reshape(np.transpose(dat5_train), newshape=[self.Nt, 2, self.Nx, self.Ny], order="F")
        dat5_train = np.transpose(np.reshape(np.squeeze(dat5_train[:, var, :, :]), newshape=[self.Nt, -1], order="F"))
        self.q_train = [dat1_train, dat2_train, dat3_train, dat4_train, dat5_train]
        self.q_polar_train = None

    def run_sPOD(self, spod_iter):
        # Reshape the variable array to suit the dimension of the input for the sPOD
        self.q_train = [np.reshape(q, newshape=[self.Nx, self.Ny, 1, self.Nt], order="F") for q in self.q_train]

        # Map the field variable from cartesian to polar coordinate system
        q_polar = []
        s0, theta_i, r_i = cartesian_to_polar(self.q_train[0], self.x, self.y, self.t)
        q_polar.append(s0)
        for samples in range(self.Nsamples_train - 1):
            s, _, _ = cartesian_to_polar(self.q_train[samples + 1], self.x, self.y, self.t)
            q_polar.append(s)

        data_shape = [self.Nx, self.Ny, 1, self.Nsamples_train * self.Nt]
        dr = r_i[1] - r_i[0]
        dtheta = theta_i[1] - theta_i[0]
        d_del = np.asarray([dr, dtheta])
        L = np.asarray([r_i[-1], theta_i[-1]])

        # Create the transformations
        trafo_train_1 = transforms(data_shape, L, shifts=self.shifts_train[0],
                                   dx=d_del,
                                   use_scipy_transform=True)
        trafo_train_2 = transforms(data_shape, L, shifts=self.shifts_train[1],
                                   trafo_type="identity", dx=d_del,
                                   use_scipy_transform=True)

        # Apply srPCA on the data
        transform_list = [trafo_train_1, trafo_train_2]
        qmat = np.concatenate([np.reshape(q, newshape=[-1, self.Nt]) for q in q_polar], axis=1)
        mu = np.prod(np.size(qmat, 0)) / (4 * np.sum(np.abs(qmat))) * 0.05
        lambd = 1 / np.sqrt(np.max([self.Nx, self.Ny]))
        ret = shifted_rPCA(qmat, transform_list, nmodes_max=100, eps=1e-4, Niter=spod_iter, use_rSVD=True, mu=mu,
                           lambd=lambd)
        sPOD_frames_train, qtilde_train, rel_err_train = ret.frames, ret.data_approx, ret.rel_err_hist
        self.q_polar_train = qmat
        ###########################################
        # %% Calculate the time amplitudes for training wildfire_data
        U_list = []
        spod_modes = []
        frame_amplitude_list_interpolation = []
        frame_amplitude_list_training = []
        cnt = 0
        for frame in sPOD_frames_train:
            VT = frame.modal_system["VT"]
            S = frame.modal_system["sigma"]
            VT = np.diag(S) @ VT
            Nmodes = frame.Nmodes
            amplitudes = [np.reshape(VT[n, :], [self.Nsamples_train, self.Nt]).T for n in range(Nmodes)]
            frame_amplitude_list_interpolation.append(amplitudes)
            frame_amplitude_list_training.append(VT)
            U_list.append(frame.modal_system["U"])
            spod_modes.append(Nmodes)
            cnt = cnt + 1

        q_spod_frames = [sPOD_frames_train[0].build_field(),
                         sPOD_frames_train[1].build_field(),
                         qtilde_train]

        return U_list, frame_amplitude_list_training, frame_amplitude_list_interpolation, spod_modes

    def plot_offline_data(self, frame_amplitude_list_training):
        plot_trainingshift(self.shifts_train, self.Nsamples_train, self.x, self.t)
        plot_timeamplitudesTraining(frame_amplitude_list_training, self.x, self.t, Nm=2)

    def test_data(self, spod_iter):
        ##########################################
        # Reshape the variable array to suit the dimension of the input for the sPOD
        q = np.reshape(self.q_test, newshape=[self.Nx, self.Ny, 1, self.Nt], order="F")

        # Map the field variable from cartesian to polar coordinate system
        q_polar, theta_i, r_i = cartesian_to_polar(q, self.x, self.y, self.t)

        # Check the transformation back and forth error between polar and cartesian coordinates (Checkpoint)
        q_cartesian = polar_to_cartesian(q_polar, self.x, self.y, theta_i, r_i, self.x_c, self.y_c, self.t)
        res = q - q_cartesian
        err = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(q, -1))
        print("Transformation back and forth error (cartesian - polar - cartesian) =  %4.4e " % err)

        data_shape = [self.Nx, self.Ny, 1, self.Nt]
        dr = r_i[1] - r_i[0]
        dtheta = theta_i[1] - theta_i[0]
        d_del = np.asarray([dr, dtheta])
        L = np.asarray([r_i[-1], theta_i[-1]])

        trafo_test_1 = transforms(data_shape, L, shifts=self.shifts_test[0],
                                  dx=d_del,
                                  use_scipy_transform=True)
        trafo_test_2 = transforms(data_shape, L, shifts=self.shifts_test[1],
                                  trafo_type="identity", dx=d_del,
                                  use_scipy_transform=True)

        # Check the transformation interpolation error
        err = give_interpolation_error(q_polar, trafo_test_1)
        print("Transformation interpolation error =  %4.4e " % err)

        ##########################################
        # %% run srPCA
        # Apply srPCA on the data
        transform_list = [trafo_test_1, trafo_test_2]
        qmat = np.reshape(q_polar, [-1, self.Nt])
        mu = np.prod(np.size(qmat, 0)) / (4 * np.sum(np.abs(qmat))) * 0.05
        lambd = 1 / np.sqrt(np.max([self.Nx, self.Ny]))
        ret = shifted_rPCA(qmat, transform_list, nmodes_max=100, eps=1e-4, Niter=spod_iter, use_rSVD=True, mu=mu,
                           lambd=lambd)
        sPOD_frames_test, qtilde_test, rel_err_test = ret.frames, ret.data_approx, ret.rel_err_hist
        self.q_polar_test = qmat

        # Deduce the frames
        modes_list = [sPOD_frames_test[0].Nmodes, sPOD_frames_test[1].Nmodes]
        q_frame_1 = sPOD_frames_test[0].build_field()
        q_frame_2 = sPOD_frames_test[1].build_field()
        qtilde = np.reshape(qtilde_test, newshape=data_shape)

        # Transform the frame wise snapshots into lab frame (moving frame)
        q_frame_1_lab = transform_list[0].apply(np.reshape(q_frame_1, newshape=data_shape))
        q_frame_2_lab = transform_list[1].apply(np.reshape(q_frame_2, newshape=data_shape))

        # Shift the pre-transformed polar data to cartesian grid to visualize
        q_frame_1_cart_lab = polar_to_cartesian(q_frame_1_lab, self.x, self.y, theta_i, r_i, self.x_c, self.y_c, self.t)
        q_frame_2_cart_lab = polar_to_cartesian(q_frame_2_lab, self.x, self.y, theta_i, r_i, self.x_c, self.y_c, self.t)
        qtilde_cart = polar_to_cartesian(qtilde, self.x, self.y, theta_i, r_i, self.x_c, self.y_c, self.t)

        # Relative reconstruction error for sPOD
        res = q - qtilde_cart
        err_full = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(q, -1))
        print("Error for full sPOD recons: {}".format(err_full))

        Q_frames_test_polar = [q_frame_1, q_frame_2, qtilde_test]
        Q_frames_test_cart = [q_frame_1_cart_lab, q_frame_2_cart_lab, qtilde_cart]

        return Q_frames_test_polar, Q_frames_test_cart

    def plot_sPOD_frames(self, Q_frames_test_cart, plot_every=10, var_name="T"):
        q = np.reshape(self.q_test, newshape=[self.Nx, self.Ny, 1, self.Nt], order="F")
        qframe0_lab = Q_frames_test_cart[0]
        qframe1_lab = Q_frames_test_cart[1]
        qtilde = Q_frames_test_cart[2]

        immpath = impath + "srPCA_2D/"
        os.makedirs(immpath, exist_ok=True)
        for n in range(self.Nt):
            if n % plot_every == 0:
                min = np.min(q[..., 0, n])
                max = np.max(q[..., 0, n])

                fig = plt.figure(figsize=(12, 5), constrained_layout=True)
                (subfig_t) = fig.subfigures(1, 1, hspace=0.05, wspace=0.1)

                gs_t = subfig_t.add_gridspec(nrows=1, ncols=6)
                ax1 = subfig_t.add_subplot(gs_t[0, 0:2])
                ax2 = subfig_t.add_subplot(gs_t[0, 2:4], sharex=ax1, sharey=ax1)
                ax3 = subfig_t.add_subplot(gs_t[0, 4:6], sharex=ax1, sharey=ax1)

                ax1.pcolormesh(self.X, self.Y, np.squeeze(qtilde[:, :, 0, n]), vmin=min, vmax=max, cmap=cmap)
                ax1.axis('scaled')
                ax1.set_title(r"sPOD")
                ax1.set_yticks([], [])
                ax1.set_xticks([], [])

                ax2.pcolormesh(self.X, self.Y, np.squeeze(qframe0_lab[:, :, 0, n]), vmin=min, vmax=max, cmap=cmap)
                ax2.axis('scaled')
                ax2.set_title(r"Frame $1$")
                ax2.set_yticks([], [])
                ax2.set_xticks([], [])

                ax3.pcolormesh(self.X, self.Y, np.squeeze(qframe1_lab[:, :, 0, n]), vmin=min, vmax=max, cmap=cmap)
                ax3.axis('scaled')
                ax3.set_title(r"Frame $2$")
                ax3.set_yticks([], [])
                ax3.set_xticks([], [])

                subfig_t.supylabel(r"space $y$")
                subfig_t.supxlabel(r"space $x$")

                fig.savefig(immpath + str(var_name) + "-" + str(n), dpi=800, transparent=True)
                plt.close(fig)

    def plot_online_data(self, frame_amplitude_predicted_sPOD, frame_amplitude_predicted_POD,
                         TA_TEST, TA_POD_TEST, TA_list_interp, shifts_predicted, SHIFTS_TEST, spod_modes,
                         U_list, U_POD_TRAIN, q_test_polar, Q_frames_test_polar):

        q1_test_polar = Q_frames_test_polar[0]
        q2_test_polar = Q_frames_test_polar[1]

        print("#############################################")
        print('Online Error checks')
        # %% Online error with respect to testing wildfire_data
        Nmf = spod_modes
        time_amplitudes_1_pred = frame_amplitude_predicted_sPOD[:Nmf[0], :]
        time_amplitudes_2_pred = frame_amplitude_predicted_sPOD[Nmf[0]:, :]
        shifts_1_pred = shifts_predicted[0, :]

        X_new = self.X - self.x_c  # Shift the origin to the center of the image
        Y_new = self.Y - self.y_c
        r = np.sqrt(X_new ** 2 + Y_new ** 2).flatten()  # polar coordinate r
        theta = np.arctan2(Y_new, X_new).flatten()  # polar coordinate theta
        r_i = np.linspace(np.min(r), np.max(r), self.Nx)
        theta_i = np.linspace(np.min(theta), np.max(theta), self.Ny)
        dr = r_i[1] - r_i[0]
        dtheta = theta_i[1] - theta_i[0]
        d_del = np.asarray([dr, dtheta])
        L = np.asarray([r_i[-1], theta_i[-1]])
        data_shape = [self.Nx, self.Ny, 1, self.Nt]
        Ndims = 2

        # %% Implement the interpolation to find the online prediction
        shifts_list_interpolated = []
        for frame in range(self.NumFrames):
            for dim in range(Ndims):
                shifts_list_interpolated.append(np.reshape(self.shifts_train[frame][dim], [self.Nsamples_train, self.Nt]).T)

        DELTA = my_delta_interpolate(shifts_list_interpolated, self.mu_vecs_train, self.mu_vecs_test)
        DELTA_PRED_FRAME_WISE = np.zeros_like(self.shifts_test)
        DELTA_PRED_FRAME_WISE[0][0] = DELTA[0]
        DELTA_PRED_FRAME_WISE[0][1] = DELTA[1]
        DELTA_PRED_FRAME_WISE[1][0] = DELTA[2]
        DELTA_PRED_FRAME_WISE[1][1] = DELTA[3]

        trafo_interpolated_1 = transforms(data_shape, L, shifts=DELTA_PRED_FRAME_WISE[0],
                                          dx=d_del,
                                          use_scipy_transform=True,
                                          interp_order=5)
        trafo_interpolated_2 = transforms(data_shape, L, shifts=DELTA_PRED_FRAME_WISE[1], trafo_type="identity",
                                          dx=d_del,
                                          use_scipy_transform=True,
                                          interp_order=5)

        trafos_interpolated = [trafo_interpolated_1, trafo_interpolated_2]

        QTILDE_FRAME_WISE, TA_INTERPOLATED = my_interpolated_state(spod_modes, U_list,
                                                                   TA_list_interp, self.mu_vecs_train,
                                                                   self.Nx, self.Ny, self.Nt, self.mu_vecs_test,
                                                                   trafos_interpolated)

        # Shifts error
        num1 = np.sqrt(np.mean(np.linalg.norm(SHIFTS_TEST.flatten() - shifts_1_pred.flatten(), 2, axis=0) ** 2))
        den1 = np.sqrt(np.mean(np.linalg.norm(SHIFTS_TEST.flatten(), 2, axis=0) ** 2))
        num1_i = np.sqrt(np.mean(np.linalg.norm(SHIFTS_TEST.flatten() - DELTA_PRED_FRAME_WISE[0][0].flatten(), 2, axis=0) ** 2))
        den1_i = np.sqrt(np.mean(np.linalg.norm(SHIFTS_TEST.flatten(), 2, axis=0) ** 2))
        print('Check 1...')
        print("Relative error indicator for shift: 1 is {}".format(num1 / den1))
        print("Relative error indicator for shift (interpolation): 1 is {}".format(num1_i / den1_i))

        # Time amplitudes error
        time_amplitudes_1_test = TA_TEST[:Nmf[0], :]
        time_amplitudes_2_test = TA_TEST[Nmf[0]:, :]
        num1 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_1_test - time_amplitudes_1_pred, 2, axis=1) ** 2))
        den1 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_1_test, 2, axis=1) ** 2))
        num2 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_2_test - time_amplitudes_2_pred, 2, axis=1) ** 2))
        den2 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_2_test, 2, axis=1) ** 2))
        num1_i = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_1_test - TA_INTERPOLATED[0], 2, axis=1) ** 2))
        den1_i = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_1_test, 2, axis=1) ** 2))
        num2_i = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_2_test - TA_INTERPOLATED[1], 2, axis=1) ** 2))
        den2_i = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_2_test, 2, axis=1) ** 2))
        num3 = np.sqrt(np.mean(np.linalg.norm(TA_POD_TEST - frame_amplitude_predicted_POD, 2, axis=1) ** 2))
        den3 = np.sqrt(np.mean(np.linalg.norm(TA_POD_TEST, 2, axis=1) ** 2))
        print('Check 2...')
        print("Relative time amplitude error (polar) indicator for frame: 1 is {}".format(num1 / den1))
        print("Relative time amplitude error (polar) indicator for frame: 2 is {}".format(num2 / den2))
        print("Relative time amplitude error (polar) indicator for frame(interpolated): 1 is {}".format(num1_i / den1_i))
        print("Relative time amplitude error (polar) indicator for frame(interpolated): 2 is {}".format(num2_i / den2_i))
        print("Relative time amplitude error (polar) indicator for POD-NN is {}".format(num3 / den3))

        # Frame wise error
        q1_pred = U_list[0] @ time_amplitudes_1_pred
        q2_pred = U_list[1] @ time_amplitudes_2_pred
        num1 = np.sqrt(np.mean(np.linalg.norm(q1_test_polar - q1_pred, 2, axis=1) ** 2))
        den1 = np.sqrt(np.mean(np.linalg.norm(q1_test_polar, 2, axis=1) ** 2))
        num2 = np.sqrt(np.mean(np.linalg.norm(q2_test_polar - q2_pred, 2, axis=1) ** 2))
        den2 = np.sqrt(np.mean(np.linalg.norm(q2_test_polar, 2, axis=1) ** 2))
        print('Check 3...')
        print("Relative frame snapshot reconstruction (polar) error indicator for frame: 1 is {}".format(num1 / den1))
        print("Relative frame snapshot reconstruction (polar) error indicator for frame: 2 is {}".format(num2 / den2))

        use_original_shift = False
        NumFrames = 2
        data_shape = [self.Nx, self.Ny, 1, self.Nt]
        Q_pred = [np.reshape(q1_pred, newshape=data_shape), np.reshape(q2_pred, newshape=data_shape)]
        q_test_polar = np.reshape(q_test_polar, newshape=data_shape)
        Q_recon_sPOD_polar = np.zeros_like(q_test_polar)
        tic = time.process_time()
        if use_original_shift:
            shifts = self.shifts_test
        else:
            shifts = self.shifts_test
            shifts[0][0] = shifts_1_pred

        trafos = [
            transforms(data_shape, L, shifts=shifts[0],
                       dx=d_del,
                       use_scipy_transform=True,
                       interp_order=5),
            transforms(data_shape, L, shifts=shifts[1],
                       trafo_type="identity", dx=d_del,
                       use_scipy_transform=True,
                       interp_order=5)
        ]

        for frame in range(NumFrames):
            Q_recon_sPOD_polar += trafos[frame].apply(Q_pred[frame])
        toc = time.process_time()
        print(f"Time consumption in sPOD DL model for full snapshot : {toc - tic:0.4f} seconds")
        res = q_test_polar - Q_recon_sPOD_polar
        err_full_sPOD = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(q_test_polar, -1))

        tic = time.process_time()
        Q_recon_POD_polar = U_POD_TRAIN @ frame_amplitude_predicted_POD
        Q_recon_POD_polar = np.reshape(Q_recon_POD_polar, newshape=data_shape)
        toc = time.process_time()
        print(f"Time consumption in POD DL model for full snapshot : {toc - tic:0.4f} seconds")
        res = q_test_polar - Q_recon_POD_polar
        err_full_POD = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(q_test_polar, -1))

        res = q_test_polar - QTILDE_FRAME_WISE
        err_full_interp = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(q_test_polar, -1))

        print('Check 4...')
        print("Relative reconstruction error indicator for full snapshot(sPOD-NN) (polar) is {}".format(err_full_sPOD))
        print("Relative reconstruction error indicator for full snapshot(POD-NN) (polar) is {}".format(err_full_POD))
        print(
            "Relative reconstruction error indicator for full snapshot(sPOD-LI) (polar) is {}".format(err_full_interp))

        num1 = np.sqrt(sum([np.square(np.linalg.norm((q_test_polar[:, :, 0, n].flatten() -
                                    Q_recon_sPOD_polar[:, :, 0, n].flatten()))) for n in range(self.Nt)]))
        den1 = np.sqrt(sum([np.square(np.linalg.norm((q_test_polar[:, :, 0, n].flatten()))) for n in range(self.Nt)]))
        num2 = np.sqrt(sum([np.square(np.linalg.norm((q_test_polar[:, :, 0, n].flatten() -
                                                      Q_recon_POD_polar[:, :, 0, n].flatten()))) for n in
                            range(self.Nt)]))
        den2 = np.sqrt(sum([np.square(np.linalg.norm((q_test_polar[:, :, 0, n].flatten()))) for n in range(self.Nt)]))
        num3 = np.sqrt(sum([np.square(np.linalg.norm((q_test_polar[:, :, 0, n].flatten() -
                                                      QTILDE_FRAME_WISE[:, :, 0, n].flatten()))) for n in
                            range(self.Nt)]))
        den3 = np.sqrt(sum([np.square(np.linalg.norm((QTILDE_FRAME_WISE[:, :, 0, n].flatten()))) for n in range(self.Nt)]))

        err_rel_sPOD = num1/den1
        err_rel_POD = num2/den2
        err_rel_interp = num3/den3
        print('Check 5...')
        print("Rel err indicator for full snapshot(sPOD-NN) (polar) is {}".format(err_rel_sPOD))
        print("Rel err indicator for full snapshot(POD-NN) (polar) is {}".format(err_rel_POD))
        print("Rel err indicator for full snapshot(sPOD-LI) (polar) is {}".format(err_rel_interp))

        num1 = [np.abs(q_test_polar[:, :, 0, n].flatten() - Q_recon_sPOD_polar[:, :, 0, n].flatten()) for n in range(self.Nt)]
        den1 = np.sqrt(sum([np.square(np.linalg.norm((q_test_polar[:, :, 0, n].flatten()))) for n in range(self.Nt)]) / self.Nt)
        num2 = [np.abs(q_test_polar[:, :, 0, n].flatten() - Q_recon_POD_polar[:, :, 0, n].flatten()) for n in range(self.Nt)]
        den2 = np.sqrt(sum([np.square(np.linalg.norm((q_test_polar[:, :, 0, n].flatten()))) for n in range(self.Nt)]) / self.Nt)
        num3 = [np.abs(q_test_polar[:, :, 0, n].flatten() - QTILDE_FRAME_WISE[:, :, 0, n].flatten()) for n in
                range(self.Nt)]
        den3 = np.sqrt(
            sum([np.square(np.linalg.norm((q_test_polar[:, :, 0, n].flatten()))) for n in range(self.Nt)]) / self.Nt)
        rel_err_sPOD_pol = [x / den1 for x in num1]
        rel_err_POD_pol = [x / den2 for x in num2]
        rel_err_interp_pol = [x / den3 for x in num3]

        # Convert the polar data into cartesian data
        Q = np.reshape(self.q_test, newshape=[self.Nx, self.Ny, 1, self.Nt], order="F")
        Q_recon_sPOD_cart = polar_to_cartesian(Q_recon_sPOD_polar, self.x, self.y, theta_i, r_i,
                                               self.x_c, self.y_c, self.t)
        Q_recon_POD_cart = polar_to_cartesian(Q_recon_POD_polar, self.x, self.y, theta_i, r_i,
                                              self.x_c, self.y_c, self.t)
        Q_recon_interp_cart = polar_to_cartesian(QTILDE_FRAME_WISE, self.x, self.y, theta_i, r_i,
                                                 self.x_c, self.y_c, self.t)
        res = Q - Q_recon_sPOD_cart
        err_full_sPOD = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(Q, -1))
        res = Q - Q_recon_POD_cart
        err_full_POD = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(Q, -1))
        res = Q - Q_recon_interp_cart
        err_full_interp = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(Q, -1))
        print('Check 6...')
        print("Relative reconstruction error indicator for full snapshot(sPOD-DL-ROM) (cartesian) is {}".format(
            err_full_sPOD))
        print(
            "Relative reconstruction error indicator for full snapshot(POD-DL-ROM) (cartesian) is {}".format(err_full_POD))

        num1 = np.sqrt(sum([np.square(np.linalg.norm((Q[:, :, 0, n].flatten() -
                                                      Q_recon_sPOD_cart[:, :, 0, n].flatten()))) for n in
                            range(self.Nt)]))
        den1 = np.sqrt(sum([np.square(np.linalg.norm((Q[:, :, 0, n].flatten()))) for n in range(self.Nt)]))
        num2 = np.sqrt(sum([np.square(np.linalg.norm((Q[:, :, 0, n].flatten() -
                                                      Q_recon_POD_cart[:, :, 0, n].flatten()))) for n in
                            range(self.Nt)]))
        den2 = np.sqrt(sum([np.square(np.linalg.norm((Q[:, :, 0, n].flatten()))) for n in range(self.Nt)]))
        num3 = np.sqrt(sum([np.square(np.linalg.norm((Q[:, :, 0, n].flatten() -
                                                      Q_recon_interp_cart[:, :, 0, n].flatten()))) for n in
                            range(self.Nt)]))
        den3 = np.sqrt(sum([np.square(np.linalg.norm((Q[:, :, 0, n].flatten()))) for n in range(self.Nt)]))
        err_rel_sPOD = num1 / den1
        err_rel_POD = num2 / den2
        err_rel_interp = num3 / den3
        print('Check 7...')
        print("Rel err indicator for full snapshot(sPOD-NN) (cartesian) is {}".format(err_rel_sPOD))
        print("Rel err indicator for full snapshot(POD-NN) (cartesian) is {}".format(err_rel_POD))
        print("Rel err indicator for full snapshot(sPOD-LI) (cartesian) is {}".format(err_rel_interp))

        num1 = [np.abs(Q[:, :, 0, n].flatten() - Q_recon_sPOD_cart[:, :, 0, n].flatten()) for n in
                range(self.Nt)]
        den1 = np.sqrt(
            sum([np.square(np.linalg.norm((Q[:, :, 0, n].flatten()))) for n in range(self.Nt)]) / self.Nt)
        num2 = [np.abs(Q[:, :, 0, n].flatten() - Q_recon_POD_cart[:, :, 0, n].flatten()) for n in
                range(self.Nt)]
        den2 = np.sqrt(
            sum([np.square(np.linalg.norm((Q[:, :, 0, n].flatten()))) for n in range(self.Nt)]) / self.Nt)
        num3 = [np.abs(Q[:, :, 0, n].flatten() - Q_recon_interp_cart[:, :, 0, n].flatten()) for n in
                range(self.Nt)]
        den3 = np.sqrt(
            sum([np.square(np.linalg.norm((Q[:, :, 0, n].flatten()))) for n in range(self.Nt)]) / self.Nt)
        rel_err_sPOD_cart = [x / den1 for x in num1]
        rel_err_POD_cart = [x / den2 for x in num2]
        rel_err_interp_cart = [x / den3 for x in num3]

        # Plot the online prediction wildfire_data
        plot_pred_comb(time_amplitudes_1_pred, time_amplitudes_1_test, time_amplitudes_2_pred,
                       time_amplitudes_2_test, TA_INTERPOLATED, shifts_1_pred, SHIFTS_TEST,
                       frame_amplitude_predicted_POD, TA_POD_TEST, self.x, self.t)
        # ###########################################################################
        # 4. in the last, implement the interpolation part
        errors = [rel_err_sPOD_pol, rel_err_POD_pol, rel_err_interp_pol, rel_err_sPOD_cart,
                  rel_err_POD_cart, rel_err_interp_cart]

        return Q_recon_sPOD_cart, Q_recon_POD_cart, Q_recon_interp_cart, errors

    def plot_recon(self, Q_recon_sPOD_cart, Q_recon_POD_cart, Q_recon_interp_cart, t_a=10, t_b=100, var_name="T"):
        q = np.reshape(self.q_test, newshape=[self.Nx, self.Ny, 1, self.Nt], order="F")

        min_a = np.min(q[..., 0, t_a-1])
        max_a = np.max(q[..., 0, t_a-1])

        min_b = np.min(q[..., 0, t_b-1])
        max_b = np.max(q[..., 0, t_b-1])

        fig = plt.figure(figsize=(10, 10), constrained_layout=True)
        (subfig_t, subfig_b) = fig.subfigures(2, 1, hspace=0.05, wspace=0.1)

        gs_t = subfig_t.add_gridspec(nrows=1, ncols=4)
        ax1 = subfig_t.add_subplot(gs_t[0, 0:2])
        ax2 = subfig_t.add_subplot(gs_t[0, 2:4], sharex=ax1, sharey=ax1)

        ax1.pcolormesh(self.X, self.Y, np.squeeze(q[:, :, 0, t_a-1]), vmin=min_a, vmax=max_a, cmap=cmap)
        ax1.axis('scaled')
        ax1.set_title(r"$t=100s$")
        ax1.set_yticks([], [])
        ax1.set_xticks([], [])
        ax1.axhline(y=self.y[self.Ny // 2 - 1], linestyle='--', color='g')
        # divider = make_axes_locatable(ax1)
        # cax = divider.append_axes('right', size='5%', pad=0.08)
        # fig.colorbar(im1, cax=cax, orientation='vertical')

        ax2.pcolormesh(self.X, self.Y, np.squeeze(q[:, :, 0, t_b-1]), vmin=min_b, vmax=max_b, cmap=cmap)
        ax2.axis('scaled')
        ax2.set_title(r"$t=1000s$")
        ax2.set_yticks([], [])
        ax2.set_xticks([], [])
        ax2.axhline(y=self.y[self.Ny // 2 - 1], linestyle='--', color='g')
        # divider = make_axes_locatable(ax2)
        # cax = divider.append_axes('right', size='5%', pad=0.08)
        # fig.colorbar(im2, cax=cax, orientation='vertical')

        subfig_t.supylabel(r"space $y$")
        subfig_t.supxlabel(r"space $x$")

        gs_b = subfig_b.add_gridspec(nrows=1, ncols=4)
        ax4 = subfig_b.add_subplot(gs_b[0, 0:2])
        ax5 = subfig_b.add_subplot(gs_b[0, 2:4], sharex=ax4)

        ax4.plot(self.x, np.squeeze(q[:, self.Ny // 2, 0, t_a-1]), color="green", linestyle="-", label='actual')
        ax4.plot(self.x, np.squeeze(Q_recon_sPOD_cart[:, self.Ny // 2, 0, t_a-1]), color="red", linestyle="--", label='sPOD-NN')
        ax4.plot(self.x, np.squeeze(Q_recon_POD_cart[:, self.Ny // 2, 0, t_a-1]), color="black", linestyle="-.", label='POD-NN')
        ax4.plot(self.x, np.squeeze(Q_recon_interp_cart[:, self.Ny // 2, 0, t_a - 1]), color="blue", linestyle="-.",
                 label='sPOD-LI')
        ax4.set_ylim(bottom=min_a - 100, top=max_a + 300)
        ax4.legend()
        ax4.grid()

        ax5.plot(self.x, np.squeeze(q[:, self.Ny // 2, 0, t_b-1]), color="green", linestyle="-", label='actual')
        ax5.plot(self.x, np.squeeze(Q_recon_sPOD_cart[:, self.Ny // 2, 0, t_b-1]), color="red", linestyle="--", label='sPOD-NN')
        ax5.plot(self.x, np.squeeze(Q_recon_POD_cart[:, self.Ny // 2, 0, t_b-1]), color="black", linestyle="-.", label='POD-NN')
        ax5.plot(self.x, np.squeeze(Q_recon_interp_cart[:, self.Ny // 2, 0, t_b - 1]), color="blue", linestyle="-.",
                 label='sPOD-LI')
        ax5.set_ylim(bottom=min_b - 100, top=max_b + 300)
        ax5.legend()
        ax5.grid()

        subfig_b.supylabel(r"$T$")
        subfig_b.supxlabel(r"space $x$")

        fig.savefig(impath + str(var_name) + "-mixed", dpi=800, transparent=True)
        plt.close(fig)


def plot_trainingshift(shifts_train, Nsamples_train, x, t):
    Nx = len(x)
    Nt = len(t)
    fig, axs = plt.subplots(1, Nsamples_train, figsize=(15, 6), num=2)
    plt.subplots_adjust(wspace=0)
    for k, ax in enumerate(axs):
        ax.plot(t, shifts_train[0][0, k * Nt:(k + 1) * Nt], color="red", marker=".", label='frame 1')
        ax.plot(t, shifts_train[1][0, k * Nt:(k + 1) * Nt], color="black", marker=".", label='frame 2')
        ax.set_title(r'${\beta}^{(' + str(k + 1) + ')}$')
        ax.grid()
        ax.legend(loc='upper right')
    fig.supxlabel(r"time $t$")
    fig.supylabel(r"space $x$")
    fig.tight_layout()
    save_fig(filepath=impath + "shifts_" + "training", figure=fig)


def plot_timeamplitudesTraining(frame_amplitude_list_training, x, t, Nm):
    Nx = len(x)
    Nt = len(t)
    fig, axs = plt.subplots(1, Nm, sharey=True, figsize=(18, 5), num=4)
    plt.subplots_adjust(wspace=0)
    for k, ax in enumerate(axs):
        ax.plot(t, frame_amplitude_list_training[0][k, :Nt], color="red",
                marker=".", label='frame 1')
        ax.plot(t, frame_amplitude_list_training[1][k, :Nt], color="black",
                marker=".", label='frame 2')
        ax.set_xticks([0, t[-1] / 2, t[-1]])
        ax.set_title(r'${mode}^{(' + str(k + 1) + ')}$')
        ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
        ax.legend(loc='upper right')
    fig.supxlabel(r"time $t$")
    fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")
    fig.tight_layout()
    save_fig(filepath=impath + "time_amplitudes_training", figure=fig)


def plot_pred_comb(time_amplitudes_1_pred, time_amplitudes_1_test, time_amplitudes_2_pred,
                   time_amplitudes_2_test, TA_interpolated, shifts_1_pred, SHIFTS_TEST, POD_frame_amplitudes_predicted,
                   TA_POD_TEST, x, t):
    Nx = len(x)
    Nt = len(t)

    fig = plt.figure(figsize=(12, 14), constrained_layout=True)
    (subfig_t) = fig.subfigures(1, 1, hspace=0.05, wspace=0.1)

    gs_t = subfig_t.add_gridspec(nrows=3, ncols=4)
    ax1 = subfig_t.add_subplot(gs_t[0, 0:2])
    ax2 = subfig_t.add_subplot(gs_t[0, 2:4], sharex=ax1)
    ax3 = subfig_t.add_subplot(gs_t[1, 0:2], sharex=ax1)
    ax4 = subfig_t.add_subplot(gs_t[1, 2:4], sharex=ax1)
    ax5 = subfig_t.add_subplot(gs_t[2, 1:3], sharex=ax1)

    ax1.plot(t, time_amplitudes_1_test[0, :Nt], color="green", linestyle='-', label='actual')
    ax1.plot(t, time_amplitudes_1_pred[0, :Nt], color="red", linestyle='--', label='sPOD-NN')
    ax1.plot(t, TA_interpolated[0][0, :Nt], color="blue", linestyle='--', label='sPOD-LI')
    ax1.set_xticks([0, t[-1] / 2, t[-1]])
    ax1.set_ylabel(r"$a_i^{k}(t,\mu)$")
    ax1.set_xticklabels([r"$0s$", r"$1000s$", r"$2000s$"])
    ax1.set_xlabel(r"(a)")
    ax1.grid()
    ax1.legend(loc='lower right')

    ax2.plot(t, time_amplitudes_2_test[0, :Nt], color="green", linestyle='-', label='actual')
    ax2.plot(t, time_amplitudes_2_pred[0, :Nt], color="red", linestyle='--', label='sPOD-NN')
    ax2.plot(t, TA_interpolated[1][0, :Nt], color="blue", linestyle='--', label='sPOD-LI')
    ax2.set_xticks([0, t[-1] / 2, t[-1]])
    ax2.set_xticklabels([r"$0s$", r"$1000s$", r"$2000s$"])
    ax2.set_xlabel(r"(b)")
    ax2.grid()
    ax2.legend(loc='upper right')

    ax3.plot(t, TA_POD_TEST[0, :], color="green", linestyle='-', label='actual')
    ax3.plot(t, POD_frame_amplitudes_predicted[0, :], color="red", linestyle='--', label='POD-NN')
    ax3.set_xticks([0, t[-1] / 2, t[-1]])
    ax3.set_ylabel(r"$a_i^{k}(t,\mu)$")
    ax3.set_xticklabels(["0s", r"$1000s$", r"$2000s$"])
    ax3.set_xlabel(r"(c)")
    ax3.legend(loc='upper right')
    ax3.grid()

    ax4.plot(t, TA_POD_TEST[1, :], color="green", linestyle='-', label='actual')
    ax4.plot(t, POD_frame_amplitudes_predicted[1, :], color="red", linestyle='--', label='POD-NN')
    ax4.set_xticks([0, t[-1] / 2, t[-1]])
    ax4.set_xticklabels(["0s", r"$1000s$", r"$2000s$"])
    ax4.set_xlabel(r"(d)")
    ax4.legend(loc='lower right')
    ax4.grid()

    ax5.plot(t, SHIFTS_TEST.flatten()[:Nt], color="green", linestyle='-', label='actual')
    ax5.plot(t, shifts_1_pred.flatten()[:Nt], color="red", linestyle='--', label='sPOD-NN')
    ax5.set_xticks([0, t[-1] / 2, t[-1]])
    ax5.set_xticklabels([r"$0s$", r"$1000s$", r"$2000s$"])
    ax5.set_ylabel(r"space $x$")
    ax5.set_xlabel(r"(e)")
    ax5.grid()
    ax5.legend(loc='upper right')

    subfig_t.supxlabel(r"time $t$")

    save_fig(filepath=impath + "all_comb_pred", figure=fig)
    fig.savefig(impath + "all_comb_pred" + ".eps", format='eps', dpi=600, transparent=True)


def cartesian_to_polar(cartesian_data, X, Y, t, method=2):

    Nx = np.size(X)
    Ny = np.size(Y)
    Nt = np.size(t)
    X_grid, Y_grid = np.meshgrid(X, Y)
    X_c = X[-1] // 2
    Y_c = Y[-1] // 2
    polar_data = np.zeros_like(cartesian_data)

    # Method 1 seems to run into problems of coordinate ordering (while correcting, look at the
    # 'F' and 'C' ordering problem)

    X_new = X_grid - X_c  # Shift the origin to the center of the image
    Y_new = Y_grid - Y_c
    r = np.sqrt(X_new ** 2 + Y_new ** 2).flatten()  # polar coordinate r
    theta = np.arctan2(Y_new, X_new).flatten()  # polar coordinate theta

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(np.min(r), np.max(r), Nx)
    theta_i = np.linspace(np.min(theta), np.max(theta), Ny)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into cartesian coordinates
    xi, yi = r_grid * np.cos(theta_grid), r_grid * np.sin(theta_grid)
    xi = xi + X_c  # Shift the origin back to the lower left corner
    yi = yi + Y_c

    if method == 1:
        from scipy.ndimage.interpolation import map_coordinates
        xi, yi = xi.flatten(), yi.flatten()
        coords = np.vstack((xi, yi))

        # Reproject the data into polar coordinates
        for k in range(Nt):
            data = map_coordinates(cartesian_data[..., 0, k], coords, order=5)
            data = np.reshape(data, newshape=[Nx, Ny])
            polar_data[..., 0, k] = data
    elif method == 2:
        from scipy.interpolate import griddata
        # Reproject the data into polar coordinates
        for k in range(Nt):
            print(k)
            data = griddata((X_grid.flatten(), Y_grid.flatten()), cartesian_data[..., 0, k].flatten('F'), (xi, yi),
                            method='cubic',
                            fill_value=0)
            data = np.reshape(data, newshape=[Nx, Ny])
            polar_data[..., 0, k] = data

    return polar_data, theta_i, r_i


def polar_to_cartesian(polar_data, X, Y, theta_i, r_i, X_c, Y_c, t, method=2):
    Nx = len(X)
    Ny = len(Y)
    Nt = len(t)
    cartesian_data = np.zeros_like(polar_data)

    # Method 1 seems to run into problems of coordinate ordering (while correcting look at the
    # 'F' and 'C' ordering problem)
    if method == 1:
        from scipy.ndimage.interpolation import map_coordinates
        # "X" and "Y" are the numpy arrays with desired cartesian coordinates, thus creating a grid
        X_grid, Y_grid = np.meshgrid(X, Y)

        # We have the "X" and "Y" coordinates of each point in the output plane thus we calculate their corresponding theta and r
        X_new = X_grid - X_c  # Shift the origin to the center of the image
        Y_new = Y_grid - Y_c
        r = np.sqrt(X_new ** 2 + Y_new ** 2).flatten()  # polar coordinate r
        theta = np.arctan2(Y_new, X_new).flatten()  # polar coordinate theta

        # Negative angles are corrected
        theta[theta < 0] = 2*np.pi + theta[theta < 0]

        # using the known theta and r steps the coordinates are mapped to those of the data grid
        dtheta = theta_i[1] - theta_i[0]
        dr = r_i[1] - r_i[0]
        theta = theta / dtheta
        r = r / dr

        # An array of polar coordinates is created
        coords = np.vstack((theta, r))

        # The data is mapped to the new coordinates
        for k in range(Nt):
            data = polar_data[:, :, 0, k]
            data = np.vstack((data, data[-1, :]))  # To avoid holes in the 360º - 0º boundary
            data = map_coordinates(data, coords, order=5, mode='constant')
            data = np.reshape(data, newshape=[Nx, Ny], order="F")
            cartesian_data[:, :, 0, k] = data
    elif method == 2:
        from scipy.interpolate import griddata
        X_grid, Y_grid = np.meshgrid(X, Y)
        X_grid, Y_grid = np.transpose(X_grid), np.transpose(Y_grid)
        # Read the polar mesh
        theta_grid, r_grid = np.meshgrid(theta_i, r_i)

        # Cartesian equivalent of polar coordinates
        xi, yi = r_grid * np.cos(theta_grid), r_grid * np.sin(theta_grid)
        xi = xi + X_c  # Shift the origin back to the lower left corner
        yi = yi + Y_c

        # Interpolate from polar to cartesian grid
        for k in range(Nt):
            print(k)
            data = polar_data[:, :, 0, k]
            data = griddata((xi.flatten(), yi.flatten()), data.flatten(), (X_grid, Y_grid),
                            method='cubic',
                            fill_value=0)
            data = np.reshape(data, newshape=[Nx, Ny])
            cartesian_data[:, :, 0, k] = data

    return cartesian_data
