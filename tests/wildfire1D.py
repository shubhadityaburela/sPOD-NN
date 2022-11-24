import time

import numpy as np
from scipy.special import eval_hermite
import os
import sys
from random import randint
from Helper import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

impath = "./plots/images_WildLandFire/"
os.makedirs(impath, exist_ok=True)

data_path = os.path.abspath("..") + '/data/'


class WildLandFire:
    def __init__(self, spod_iter, var):
        # dat1_train = np.load(data_path + 'SnapShotMatrix540.npy')
        # dat2_train = np.load(data_path + 'SnapShotMatrix560.npy')
        # dat3_train = np.load(data_path + 'SnapShotMatrix580.npy')
        # self.x = np.load(data_path + 'Grid.npy')
        # self.t = np.load(data_path + 'Time.npy')
        # delta1_train = np.load(data_path + 'Shifts540.npy')
        # delta2_train = np.load(data_path + 'Shifts560.npy')
        # delta3_train = np.load(data_path + 'Shifts580.npy')
        #
        # self.Nx = np.size(self.x)
        # self.Nt = np.size(self.t)
        # q_train_total = np.concatenate((dat1_train, dat2_train, dat3_train), axis=1)
        # q_train = q_train_total[var * self.Nx:(var + 1) * self.Nx, :]
        # shifts_train = np.concatenate((delta1_train, delta2_train, delta3_train), axis=1)
        #
        # Nsamples_train = 3
        # self.NumFrames = 3
        #
        # self.mu_vecs_train = np.asarray([540, 560, 580])
        # params_train = [np.squeeze(np.asarray([[np.ones_like(self.t) * mu], [self.t]])) for mu in self.mu_vecs_train]
        # params_train = np.concatenate(params_train, axis=1)
        #
        # self.params_instance = len(self.mu_vecs_train)

        dat1_train = np.load(data_path + 'SnapShotMatrix540.npy')
        dat2_train = np.load(data_path + 'SnapShotMatrix545.npy')
        dat3_train = np.load(data_path + 'SnapShotMatrix550.npy')
        dat4_train = np.load(data_path + 'SnapShotMatrix555.npy')
        dat5_train = np.load(data_path + 'SnapShotMatrix560.npy')
        dat6_train = np.load(data_path + 'SnapShotMatrix565.npy')
        dat7_train = np.load(data_path + 'SnapShotMatrix570.npy')
        dat8_train = np.load(data_path + 'SnapShotMatrix575.npy')
        dat9_train = np.load(data_path + 'SnapShotMatrix580.npy')
        self.x = np.load(data_path + 'Grid.npy')
        self.t = np.load(data_path + 'Time.npy')
        delta1_train = np.load(data_path + 'Shifts540.npy')
        delta2_train = np.load(data_path + 'Shifts545.npy')
        delta3_train = np.load(data_path + 'Shifts550.npy')
        delta4_train = np.load(data_path + 'Shifts555.npy')
        delta5_train = np.load(data_path + 'Shifts560.npy')
        delta6_train = np.load(data_path + 'Shifts565.npy')
        delta7_train = np.load(data_path + 'Shifts570.npy')
        delta8_train = np.load(data_path + 'Shifts575.npy')
        delta9_train = np.load(data_path + 'Shifts580.npy')

        # Trimming the data to half
        self.t = self.t[:len(self.t) // 2]
        self.Nt = np.size(self.t)
        dat1_train = dat1_train[:, :self.Nt]
        dat2_train = dat2_train[:, :self.Nt]
        dat3_train = dat3_train[:, :self.Nt]
        dat4_train = dat4_train[:, :self.Nt]
        dat5_train = dat5_train[:, :self.Nt]
        dat6_train = dat6_train[:, :self.Nt]
        dat7_train = dat7_train[:, :self.Nt]
        dat8_train = dat8_train[:, :self.Nt]
        dat9_train = dat9_train[:, :self.Nt]

        delta1_train = delta1_train[:, :self.Nt]
        delta2_train = delta2_train[:, :self.Nt]
        delta3_train = delta3_train[:, :self.Nt]
        delta4_train = delta4_train[:, :self.Nt]
        delta5_train = delta5_train[:, :self.Nt]
        delta6_train = delta6_train[:, :self.Nt]
        delta7_train = delta7_train[:, :self.Nt]
        delta8_train = delta8_train[:, :self.Nt]
        delta9_train = delta9_train[:, :self.Nt]

        self.Nx = np.size(self.x)
        q_train_total = np.concatenate((dat1_train, dat2_train, dat3_train, dat4_train, dat5_train,
                                        dat6_train, dat7_train, dat8_train, dat9_train), axis=1)
        q_train = q_train_total[var * self.Nx:(var + 1) * self.Nx, :]
        shifts_train = np.concatenate((delta1_train, delta2_train, delta3_train, delta4_train, delta5_train,
                                       delta6_train, delta7_train, delta8_train, delta9_train), axis=1)

        Nsamples_train = 9
        self.NumFrames = 3

        self.mu_vecs_train = np.asarray([540, 545, 550, 555, 560, 565, 570, 575, 580])
        params_train = [np.squeeze(np.asarray([[np.ones_like(self.t) * mu], [self.t]])) for mu in self.mu_vecs_train]
        params_train = np.concatenate(params_train, axis=1)

        self.params_instance = len(self.mu_vecs_train)

        print("#############################################")
        print("Data checks....")
        ##########################################
        # %% Run srPCA
        self.dx = self.x[1] - self.x[0]
        self.L = [self.x[-1]]
        data_shape = [self.Nx, 1, 1, self.Nt * Nsamples_train]
        trafo_train_1 = transforms(data_shape, self.L + self.dx, shifts=np.squeeze(shifts_train[0]).flatten(),
                                   dx=[self.dx],
                                   use_scipy_transform=False,
                                   interp_order=5)
        trafo_train_2 = transforms(data_shape, self.L + self.dx, shifts=np.squeeze(shifts_train[1]).flatten(),
                                   trafo_type="identity", dx=[self.dx],
                                   use_scipy_transform=False,
                                   interp_order=5)
        trafo_train_3 = transforms(data_shape, self.L + self.dx, shifts=np.squeeze(shifts_train[2]).flatten(),
                                   dx=[self.dx],
                                   use_scipy_transform=False,
                                   interp_order=5)
        trafos_train = [trafo_train_1, trafo_train_2, trafo_train_3]

        qmat = np.reshape(q_train, [-1, self.Nt * Nsamples_train])
        [N, M] = np.shape(qmat)
        mu0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.005
        lambd0 = 1 / np.sqrt(np.maximum(M, N)) * 10

        ret_train = shifted_rPCA(q_train, trafos_train, nmodes_max=60, eps=1e-16, Niter=spod_iter, use_rSVD=True,
                                   mu=mu0, lambd=lambd0, dtol=1e-5)
        sPOD_frames_train, qtilde_train, rel_err_train = ret_train.frames, ret_train.data_approx, ret_train.rel_err_hist

        ###########################################
        # %% relative offline error for training data (srPCA error)
        err_full = np.sqrt(np.mean(np.linalg.norm(q_train - qtilde_train, 2, axis=1) ** 2)) / \
                     np.sqrt(np.mean(np.linalg.norm(q_train, 2, axis=1) ** 2))
        print("Check 1...")
        print("Error for full sPOD recons: {}".format(err_full))

        ###########################################
        # %% Calculate the time amplitudes for training data
        self.U_list = []
        self.spodModes = []
        self.frame_amplitude_list = []
        frame_amplitude_list_training = []
        self.shifts_list = []
        cnt = 0
        for frame in sPOD_frames_train:
            VT = frame.modal_system["VT"]
            S = frame.modal_system["sigma"]
            VT = np.diag(S) @ VT
            Nmodes = frame.Nmodes
            amplitudes = [np.reshape(VT[n, :], [Nsamples_train, self.Nt]).T for n in range(Nmodes)]
            shifts = np.reshape(shifts_train[cnt], [Nsamples_train, self.Nt]).T
            self.frame_amplitude_list.append(amplitudes)
            frame_amplitude_list_training.append(VT)
            self.U_list.append(frame.modal_system["U"])
            self.spodModes.append(Nmodes)
            self.shifts_list.append(shifts)
            cnt = cnt + 1

        q1_spod_frame = sPOD_frames_train[0].build_field()
        q2_spod_frame = sPOD_frames_train[1].build_field()
        q3_spod_frame = sPOD_frames_train[2].build_field()

        ###########################################
        # %% Generate data for the POD-DL-ROM for comparison
        U, S, VT = np.linalg.svd(np.squeeze(q_train), full_matrices=False)
        self.U_POD_TRAIN = U[:, :sum(self.spodModes) + 2]
        self.TA_POD_TRAIN = np.diag(S[:sum(self.spodModes) + 2]) @ VT[:sum(self.spodModes) + 2, :]

        ###########################################
        # %% data for the NN
        amplitudes_train = np.concatenate(frame_amplitude_list_training, axis=0)
        self.TA_TRAIN = amplitudes_train
        self.SHIFTS_TRAIN = [shifts_train[0], shifts_train[2]]
        self.PARAMS_TRAIN = params_train

        ###########################################
        # %% Plot all the variables required
        plot_trainingdata(q_train, Nsamples_train, self.x, self.t)
        plot_trainingshift(shifts_train, Nsamples_train, self.x, self.t)
        plot_sPODframes(q_train, q1_spod_frame, q2_spod_frame, q3_spod_frame, qtilde_train, self.x, self.t)
        plot_timeamplitudesTraining(frame_amplitude_list_training, self.x, self.t, Nm=2)


def WildLandFire_testing(Q, delta, U_list, U_POD_TRAIN, x, t, val, var, spod_iter):
    Nx = len(x)
    Nt = len(t)

    q_test = Q[var * Nx:(var + 1) * Nx, :]
    shifts_test = delta

    mu_vecs_test = np.asarray([val])
    params_test = [np.squeeze(np.asarray([[np.ones_like(t) * mu], [t]])) for mu in mu_vecs_test]
    params_test = np.concatenate(params_test, axis=1)

    TA_POD_TEST = U_POD_TRAIN.transpose() @ q_test

    ##########################################
    # %% Calculate the transformation interpolation error
    dat = q_test
    data_shape = [Nx, 1, 1, Nt]
    dx = x[1] - x[0]
    L = [x[-1]]
    q = np.reshape(dat, data_shape)

    trafo_test_1 = transforms(data_shape, L + dx, shifts=shifts_test[0], dx=[dx],
                              use_scipy_transform=False,
                              interp_order=5)
    trafo_test_2 = transforms(data_shape, L + dx, shifts=shifts_test[1], trafo_type="identity",
                              dx=[dx],
                              use_scipy_transform=False, interp_order=5)
    trafo_test_3 = transforms(data_shape, L + dx, shifts=shifts_test[2], dx=[dx],
                              use_scipy_transform=False,
                              interp_order=5)

    interp_err = give_interpolation_error(q, trafo_test_1)
    print("Transformation interpolation error =  %4.4e " % interp_err)

    ##########################################
    # %% run srPCA
    trafos_test = [trafo_test_1, trafo_test_2, trafo_test_3]

    qmat = np.reshape(q_test, [-1, Nt])
    [N, M] = np.shape(qmat)
    mu0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.01
    lambd0 = 1 / np.sqrt(np.maximum(M, N)) * 10

    ret_test = shifted_rPCA(q_test, trafos_test, nmodes_max=60, eps=1e-16, Niter=spod_iter, use_rSVD=True,
                            mu=mu0, lambd=lambd0, dtol=1e-5)
    sPOD_frames_test, qtilde_test, rel_err_test = ret_test.frames, ret_test.data_approx, ret_test.rel_err_hist

    q1_test = sPOD_frames_test[0].build_field()
    q2_test = sPOD_frames_test[1].build_field()
    q3_test = sPOD_frames_test[2].build_field()

    ###########################################
    # %% Data for NN prediction
    time_amplitudes_1_test = U_list[0].transpose() @ q1_test
    time_amplitudes_2_test = U_list[1].transpose() @ q2_test
    time_amplitudes_3_test = U_list[2].transpose() @ q3_test
    amplitudes_test = np.concatenate(
        (time_amplitudes_1_test, time_amplitudes_2_test, time_amplitudes_3_test), axis=0)

    TA_TEST = amplitudes_test
    SHIFTS_TEST = [shifts_test[0], shifts_test[2]]
    PARAMS_TEST = params_test

    return TA_TEST, TA_POD_TEST, SHIFTS_TEST, PARAMS_TEST, mu_vecs_test, q_test, q1_test, q2_test, q3_test


def onlineErroranalysis(frame_amplitude_predicted, TA_TEST, TA_POD_TEST, frame_amplitude_list, shifts_predicted,
                        SHIFTS_TEST, shifts_list, POD_frame_amplitudes_predicted, spodModes, U_list, U_POD_TRAIN,
                        q_test, q1_test, q2_test, q3_test, mu_vecs_train, mu_vecs_test, x, t):
    print("#############################################")
    print('Online Error checks')
    # %% Online error with respect to testing data
    Nx = len(x)
    Nt = len(t)
    dx = x[1] - x[0]
    Nmf = spodModes
    time_amplitudes_1_pred = frame_amplitude_predicted[:Nmf[0], :]
    time_amplitudes_2_pred = frame_amplitude_predicted[Nmf[0]:Nmf[0] + Nmf[1], :]
    time_amplitudes_3_pred = frame_amplitude_predicted[Nmf[0] + Nmf[1]:, :]
    shifts_1_pred = shifts_predicted[0, :]
    shifts_3_pred = shifts_predicted[1, :]

    # %% Implement the interpolation to find the online prediction
    tic = time.process_time()
    DELTA_PRED_FRAME_WISE = my_delta_interpolate(shifts_list, mu_vecs_train, mu_vecs_test)
    data_shape = [Nx, 1, 1, Nt]
    L = [x[-1]]

    trafo_interpolated_1 = transforms(data_shape, L + dx, shifts=DELTA_PRED_FRAME_WISE[0], dx=[dx],
                                      use_scipy_transform=False,
                                      interp_order=5)
    trafo_interpolated_2 = transforms(data_shape, L + dx, shifts=DELTA_PRED_FRAME_WISE[1], trafo_type="identity",
                                      dx=[dx],
                                      use_scipy_transform=False, interp_order=5)
    trafo_interpolated_3 = transforms(data_shape, L + dx, shifts=DELTA_PRED_FRAME_WISE[2], dx=[dx],
                                      use_scipy_transform=False,
                                      interp_order=5)

    trafos_interpolated = [trafo_interpolated_1, trafo_interpolated_2, trafo_interpolated_3]

    QTILDE_FRAME_WISE, TA_INTERPOLATED = my_interpolated_state(spodModes, U_list,
                                                               frame_amplitude_list, mu_vecs_train,
                                                               Nx, Nt, mu_vecs_test, trafos_interpolated)
    toc = time.process_time()
    print(f"Time consumption in interpolation for full snapshot : {toc - tic:0.4f} seconds")
    # Shifts error
    num1_i = np.sqrt(np.mean(np.linalg.norm(SHIFTS_TEST[0] - DELTA_PRED_FRAME_WISE[0], 2, axis=0) ** 2))
    den1_i = np.sqrt(np.mean(np.linalg.norm(SHIFTS_TEST[0], 2, axis=0) ** 2))
    num3_i = np.sqrt(np.mean(np.linalg.norm(SHIFTS_TEST[1] - DELTA_PRED_FRAME_WISE[2], 2, axis=0) ** 2))
    den3_i = np.sqrt(np.mean(np.linalg.norm(SHIFTS_TEST[1], 2, axis=0) ** 2))

    num1 = np.sqrt(np.mean(np.linalg.norm(SHIFTS_TEST[0] - shifts_1_pred.flatten(), 2, axis=0) ** 2))
    den1 = np.sqrt(np.mean(np.linalg.norm(SHIFTS_TEST[0], 2, axis=0) ** 2))
    num3 = np.sqrt(np.mean(np.linalg.norm(SHIFTS_TEST[1] - shifts_3_pred.flatten(), 2, axis=0) ** 2))
    den3 = np.sqrt(np.mean(np.linalg.norm(SHIFTS_TEST[1], 2, axis=0) ** 2))
    print('Check 1...')
    print("Relative error indicator for shift: 1 is {}".format(num1 / den1))
    print("Relative error indicator for shift: 3 is {}".format(num3 / den3))
    print("Relative error indicator for shift(interpolated): 1 is {}".format(num1_i / den1_i))
    print("Relative error indicator for shift(interpolated): 3 is {}".format(num3_i / den3_i))

    # Time amplitudes error
    time_amplitudes_1_test = TA_TEST[:Nmf[0], :]
    time_amplitudes_2_test = TA_TEST[Nmf[0]:Nmf[0] + Nmf[1], :]
    time_amplitudes_3_test = TA_TEST[Nmf[0] + Nmf[1]:, :]
    num1 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_1_test - time_amplitudes_1_pred, 2, axis=1) ** 2))
    den1 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_1_test, 2, axis=1) ** 2))
    num2 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_2_test - time_amplitudes_2_pred, 2, axis=1) ** 2))
    den2 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_2_test, 2, axis=1) ** 2))
    num3 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_3_test - time_amplitudes_3_pred, 2, axis=1) ** 2))
    den3 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_3_test, 2, axis=1) ** 2))

    num4 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_1_test - TA_INTERPOLATED[0], 2, axis=1) ** 2))
    den4 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_1_test, 2, axis=1) ** 2))
    num5 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_2_test - TA_INTERPOLATED[1], 2, axis=1) ** 2))
    den5 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_2_test, 2, axis=1) ** 2))
    num6 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_3_test - TA_INTERPOLATED[2], 2, axis=1) ** 2))
    den6 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_3_test, 2, axis=1) ** 2))

    num7 = np.sqrt(np.mean(np.linalg.norm(TA_POD_TEST - POD_frame_amplitudes_predicted, 2, axis=1) ** 2))
    den7 = np.sqrt(np.mean(np.linalg.norm(TA_POD_TEST, 2, axis=1) ** 2))
    print('Check 2...')
    print("Relative time amplitude error indicator for frame: 1 is {}".format(num1 / den1))
    print("Relative time amplitude error indicator for frame: 2 is {}".format(num2 / den2))
    print("Relative time amplitude error indicator for frame: 3 is {}".format(num3 / den3))
    print("Relative time amplitude error indicator (interpolated) for frame: 1 is {}".format(num4 / den4))
    print("Relative time amplitude error indicator (interpolated) for frame: 2 is {}".format(num5 / den5))
    print("Relative time amplitude error indicator (interpolated) for frame: 3 is {}".format(num6 / den6))
    print("Relative time amplitude error indicator for POD-DL-ROM is {}".format(num7 / den7))

    # Frame wise error
    q1_pred = U_list[0] @ time_amplitudes_1_pred
    q2_pred = U_list[1] @ time_amplitudes_2_pred
    q3_pred = U_list[2] @ time_amplitudes_3_pred
    num1 = np.sqrt(np.mean(np.linalg.norm(q1_test - q1_pred, 2, axis=1) ** 2))
    den1 = np.sqrt(np.mean(np.linalg.norm(q1_test, 2, axis=1) ** 2))
    num2 = np.sqrt(np.mean(np.linalg.norm(q2_test - q2_pred, 2, axis=1) ** 2))
    den2 = np.sqrt(np.mean(np.linalg.norm(q2_test, 2, axis=1) ** 2))
    num3 = np.sqrt(np.mean(np.linalg.norm(q3_test - q3_pred, 2, axis=1) ** 2))
    den3 = np.sqrt(np.mean(np.linalg.norm(q3_test, 2, axis=1) ** 2))
    print('Check 3...')
    print("Relative frame snapshot reconstruction error indicator for frame: 1 is {}".format(num1 / den1))
    print("Relative frame snapshot reconstruction error indicator for frame: 2 is {}".format(num2 / den2))
    print("Relative frame snapshot reconstruction error indicator for frame: 3 is {}".format(num3 / den3))

    # Total reconstructed error
    num1_i = np.sqrt(np.mean(np.linalg.norm(q_test - QTILDE_FRAME_WISE, 2, axis=1) ** 2))
    den1_i = np.sqrt(np.mean(np.linalg.norm(q_test, 2, axis=1) ** 2))

    use_original_shift = False
    Q_recon_sPOD = np.zeros_like(q1_pred)
    NumFrames = 3
    Q_pred = [q1_pred, q2_pred, q3_pred]
    if use_original_shift:
        print("Not implemented yet")
        exit()
    else:
        tic = time.process_time()
        shifts_2_pred = np.zeros_like(shifts_1_pred)
        L = [x[-1]]
        trafos = [
            transforms([Nx, 1, 1, Nt], L + dx, shifts=np.squeeze(shifts_1_pred),
                       dx=[dx],
                       use_scipy_transform=False,
                       interp_order=5),
            transforms([Nx, 1, 1, Nt], L + dx, shifts=np.squeeze(shifts_2_pred), trafo_type="identity",
                       dx=[dx],
                       use_scipy_transform=False,
                       interp_order=5),
            transforms([Nx, 1, 1, Nt], L + dx, shifts=np.squeeze(shifts_3_pred),
                       dx=[dx],
                       use_scipy_transform=False,
                       interp_order=5)
        ]
        for frame in range(NumFrames):
            Q_recon_sPOD += trafos[frame].apply(Q_pred[frame])
        toc = time.process_time()
        print(f"Time consumption in sPOD DL model for full snapshot : {toc - tic:0.4f} seconds")

    num1 = np.sqrt(np.mean(np.linalg.norm(q_test - Q_recon_sPOD, 2, axis=1) ** 2))
    den1 = np.sqrt(np.mean(np.linalg.norm(q_test, 2, axis=1) ** 2))

    tic = time.process_time()
    Q_recon_POD = U_POD_TRAIN @ POD_frame_amplitudes_predicted
    toc = time.process_time()
    print(f"Time consumption in POD DL model for full snapshot : {toc - tic:0.4f} seconds")

    num2 = np.sqrt(np.mean(np.linalg.norm(q_test - Q_recon_POD, 2, axis=1) ** 2))
    den2 = np.sqrt(np.mean(np.linalg.norm(q_test, 2, axis=1) ** 2))
    print('Check 4...')
    print("Relative reconstruction error indicator for full snapshot(sPOD-DL-ROM) is {}".format(num1 / den1))
    print("Relative reconstruction error indicator for full snapshot(POD-DL-ROM) is {}".format(num2 / den2))
    print("Relative reconstruction error indicator for full snapshot(interpolated) is {}".format(num1_i / den1_i))

    # Plot the online prediction data
    Nm = 2
    plot_timeamplitudesPred(time_amplitudes_1_pred, time_amplitudes_1_test, time_amplitudes_2_pred,
                            time_amplitudes_2_test, time_amplitudes_3_pred, time_amplitudes_3_test,
                            TA_INTERPOLATED, x, t, Nm)
    plot_timeamplitudes_shifts_pred(time_amplitudes_1_pred, time_amplitudes_1_test, time_amplitudes_2_pred,
                            time_amplitudes_2_test, time_amplitudes_3_pred, time_amplitudes_3_test,
                            TA_INTERPOLATED, shifts_1_pred, shifts_3_pred, SHIFTS_TEST, DELTA_PRED_FRAME_WISE,
                                    x, t)
    plot_timeamplitudesPredPOD(POD_frame_amplitudes_predicted, TA_POD_TEST, x, t, nmodes=5)
    plot_timeamplitudesPredPOD_few(POD_frame_amplitudes_predicted, TA_POD_TEST, x, t)
    plot_shiftsPred(shifts_1_pred, shifts_3_pred, SHIFTS_TEST, DELTA_PRED_FRAME_WISE, x, t)
    plot_recons_snapshot(q_test, QTILDE_FRAME_WISE, Q_recon_sPOD, Q_recon_POD, x, t)
    plot_recons_snapshot_cross_section(q_test, QTILDE_FRAME_WISE, Q_recon_sPOD, Q_recon_POD, x, t)


    return Q_recon_sPOD


def plot_trainingdata(q_train, Nsamples_train, x, t):
    Nx = len(x)
    Nt = len(t)

    [Xgrid, Tgrid] = meshgrid(x, t)
    Xgrid = Xgrid.T
    Tgrid = Tgrid.T

    fig, axs = plt.subplots(1, Nsamples_train, figsize=(15, 6), sharey=True, sharex=True, num=1)
    plt.subplots_adjust(wspace=0.25)
    qmin = np.min(q_train)
    qmax = np.max(q_train)
    for k in range(0, Nsamples_train):
        kw = k
        axs[kw].pcolormesh(Xgrid, Tgrid, q_train[:, Nt * k:Nt * (k + 1)], vmin=qmin, vmax=qmax)
        axs[kw].set_title(r'${\beta}^{(' + str(k + 1) + ')}$')
        axs[kw].set_yticks([], [])
        axs[kw].set_xticks([], [])

    # fig.supxlabel(r"$x$")
    # fig.supylabel(r"$t$")
    fig.subplots_adjust(right=0.8)
    save_fig(filepath=impath + "Snapshot_training", figure=fig)


def plot_trainingshift(shifts_train, Nsamples_train, x, t):
    Nx = len(x)
    Nt = len(t)
    fig, axs = plt.subplots(1, Nsamples_train, figsize=(15, 6), num=2)
    plt.subplots_adjust(wspace=0)
    for k, ax in enumerate(axs):
        ax.plot(t, shifts_train[0, k * Nt:(k + 1) * Nt], color="red", marker=".", label='frame 1')
        ax.plot(t, shifts_train[1, k * Nt:(k + 1) * Nt], color="black", marker=".", label='frame 2')
        ax.plot(t, shifts_train[2, k * Nt:(k + 1) * Nt], color="blue", marker=".", label='frame 3')
        ax.set_title(r'${\beta}^{(' + str(k + 1) + ')}$')
        ax.grid()
        ax.legend(loc='upper right')
    fig.supxlabel(r"time $t$")
    fig.supylabel(r"space $x$")
    fig.tight_layout()
    save_fig(filepath=impath + "shifts_" + "training", figure=fig)


def plot_sPODframes(q_train, q1_spod_frame, q2_spod_frame, q3_spod_frame, qtilde, x, t):
    Nx = len(x)
    Nt = len(t)

    [Xgrid, Tgrid] = meshgrid(x, t)
    Xgrid = Xgrid.T
    Tgrid = Tgrid.T

    qmin = np.min(q_train)
    qmax = np.max(q_train)
    fig, axs = plt.subplots(1, 4, num=3, sharey=True, figsize=(15, 6))
    # 1. frame
    k_frame = 0
    axs[0].pcolormesh(Xgrid, Tgrid, q1_spod_frame[:, :Nt], vmin=qmin, vmax=qmax)
    # axs[0].set_title(r"$Q^" + str(k_frame + 1) + "_{i,j}$")
    axs[0].set_yticks([], [])
    axs[0].set_xticks([], [])
    # 2. frame
    k_frame = 1
    axs[1].pcolormesh(Xgrid, Tgrid, q2_spod_frame[:, :Nt], vmin=qmin, vmax=qmax)
    # axs[1].set_title(r"$Q^" + str(k_frame + 1) + "_{i,j}$")
    axs[1].set_yticks([], [])
    axs[1].set_xticks([], [])
    # 3. frame
    k_frame = 2
    axs[2].pcolormesh(Xgrid, Tgrid, q3_spod_frame[:, :Nt], vmin=qmin, vmax=qmax)
    # axs[2].set_title(r"$Q^" + str(k_frame + 1) + "_{i,j}$")
    axs[2].set_yticks([], [])
    axs[2].set_xticks([], [])
    # Reconstruction
    axs[3].pcolormesh(Xgrid, Tgrid, qtilde[:, :Nt], vmin=qmin, vmax=qmax)
    # axs[3].set_title(r"$\tilde{Q}" + "_{i,j}$")
    axs[3].set_yticks([], [])
    axs[3].set_xticks([], [])
    plt.tight_layout()

    save_fig(filepath=impath + "frames_sPOD", figure=fig)


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
        ax.plot(t, frame_amplitude_list_training[2][k, :Nt], color="blue",
                marker=".", label='frame 3')
        ax.set_xticks([0, t[-1] / 2, t[-1]])
        ax.set_title(r'${mode}^{(' + str(k + 1) + ')}$')
        ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
        ax.legend(loc='upper right')
    fig.supxlabel(r"time $t$")
    fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")
    fig.tight_layout()
    save_fig(filepath=impath + "time_amplitudes_training", figure=fig)


def plot_timeamplitudesPred(time_amplitudes_1_pred, time_amplitudes_1_test, time_amplitudes_2_pred,
                            time_amplitudes_2_test, time_amplitudes_3_pred, time_amplitudes_3_test,
                            TA_interpolated, x, t, Nm):
    Nx = len(x)
    Nt = len(t)
    # Frame 1
    fig, axs = plt.subplots(1, Nm, sharey=True, figsize=(18, 5), num=5)
    plt.subplots_adjust(wspace=0)
    for k, ax in enumerate(axs):
        ax.plot(t, time_amplitudes_1_pred[k, :], color="red", marker=".", label='predicted')
        ax.plot(t, time_amplitudes_1_test[k, :], color="blue", marker=".", label='actual')
        ax.plot(t, TA_interpolated[0][k, :], color="yellow", marker=".", label='interpolated')
        ax.set_xticks([0, t[-1] / 2, t[-1]])
        ax.set_title(r'${mode}^{(' + str(k + 1) + ')}$')
        ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
        ax.legend(loc='upper right')
    fig.supxlabel(r"time $t$")
    fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")
    fig.tight_layout()
    save_fig(filepath=impath + "time_amplitudes_frame_1_predicted", figure=fig)

    # Frame 2
    fig, axs = plt.subplots(1, Nm, sharey=True, figsize=(18, 5), num=6)
    plt.subplots_adjust(wspace=0)
    for k, ax in enumerate(axs):
        ax.plot(t, time_amplitudes_2_pred[k, :], color="red", marker=".", label='predicted')
        ax.plot(t, time_amplitudes_2_test[k, :], color="blue", marker=".", label='actual')
        ax.plot(t, TA_interpolated[1][k, :], color="yellow", marker=".", label='interpolated')
        ax.set_xticks([0, t[-1] / 2, t[-1]])
        ax.set_title(r'${mode}^{(' + str(k + 1) + ')}$')
        ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
        ax.legend(loc='upper right')
    fig.supxlabel(r"time $t$")
    fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")
    fig.tight_layout()
    save_fig(filepath=impath + "time_amplitudes_frame_2_predicted", figure=fig)

    # Frame 3
    fig, axs = plt.subplots(1, Nm, sharey=True, figsize=(18, 5), num=7)
    plt.subplots_adjust(wspace=0)
    for k, ax in enumerate(axs):
        ax.plot(t, time_amplitudes_3_pred[k, :], color="red", marker=".", label='predicted')
        ax.plot(t, time_amplitudes_3_test[k, :], color="blue", marker=".", label='actual')
        ax.plot(t, TA_interpolated[2][k, :], color="yellow", marker=".", label='interpolated')
        ax.set_xticks([0, t[-1] / 2, t[-1]])
        ax.set_title(r'${mode}^{(' + str(k + 1) + ')}$')
        ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
        ax.legend(loc='upper right')
    fig.supxlabel(r"time $t$")
    fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")
    fig.tight_layout()
    save_fig(filepath=impath + "time_amplitudes_frame_3_predicted", figure=fig)


def plot_timeamplitudes_shifts_pred(time_amplitudes_1_pred, time_amplitudes_1_test, time_amplitudes_2_pred,
                            time_amplitudes_2_test, time_amplitudes_3_pred, time_amplitudes_3_test,
                            TA_interpolated, shifts_1_pred, shifts_3_pred, SHIFTS_TEST, delta_pred_frame_wise, x, t):
    Nx = len(x)
    Nt = len(t)
    t = t[:Nt//2]
    # Time amplitudes for all frames
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), num=20)
    plt.subplots_adjust(wspace=1)
    axs[0].plot(t, time_amplitudes_1_test[0, :Nt//2], color="green", linestyle='-', label='actual')
    axs[0].plot(t, time_amplitudes_1_pred[0, :Nt//2], color="red", linestyle='--', label='NN')
    axs[0].plot(t, TA_interpolated[0][0, :Nt//2], color="blue", linestyle='--', label='interpolated')
    axs[0].set_xticks([0, t[-1]/2, t[-1]])
    # axs[0].set_ylabel(r"$a_i^{k}(t,\mu)$")
    # axs[0].set_title(r'${mode}^{(' + str(1) + ')}$')
    axs[0].set_xticklabels(["0s", r"$350s$", r"$700s$"])
    axs[0].grid()
    axs[0].legend(loc='lower right')

    axs[1].plot(t, time_amplitudes_2_test[0, :Nt//2], color="green", linestyle='-', label='actual')
    axs[1].plot(t, time_amplitudes_2_pred[0, :Nt//2], color="red", linestyle='--', label='NN')
    axs[1].plot(t, TA_interpolated[1][0, :Nt//2], color="blue", linestyle='--', label='interpolated')
    axs[1].set_xticks([0, t[-1]/2, t[-1]])
    # axs[1].set_ylabel(r"$a_i^{k}(t,\mu)$")
    # axs[1].set_title(r'${mode}^{(' + str(1) + ')}$')
    axs[1].set_xticklabels(["0s", r"$350s$", r"$700s$"])
    axs[1].grid()
    axs[1].legend(loc='upper right')

    axs[2].plot(t, time_amplitudes_3_test[0, :Nt//2], color="green", linestyle='-', label='actual')
    axs[2].plot(t, time_amplitudes_3_pred[0, :Nt//2], color="red", linestyle='--', label='NN')
    axs[2].plot(t, TA_interpolated[2][0, :Nt//2], color="blue", linestyle='--', label='interpolated')
    axs[2].set_xticks([0, t[-1]/2, t[-1]])
    # axs[2].set_ylabel(r"$a_i^{k}(t,\mu)$")
    # axs[2].set_title(r'${mode}^{(' + str(1) + ')}$')
    axs[2].set_xticklabels(["0s", r"$350s$", r"$700s$"])
    axs[2].grid()
    axs[2].legend(loc='lower right')

    # fig.supxlabel(r"time $t$")
    fig.supylabel(r"$a_i^{k}(t,\mu)$")
    fig.tight_layout()
    save_fig(filepath=impath + "time_amplitudes_sPOD_predicted", figure=fig)


    # Shifts for all the frames
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), num=21)
    plt.subplots_adjust(wspace=1)
    axs[0].plot(t, SHIFTS_TEST[0].flatten()[:Nt//2], color="green", linestyle='-', label='actual')
    axs[0].plot(t, shifts_1_pred.flatten()[:Nt//2], color="red", linestyle='--', label='NN')
    axs[0].plot(t, delta_pred_frame_wise[0][:Nt//2], color="blue", linestyle='--', label='interpolated')
    axs[0].set_xticks([0, t[-1] / 2, t[-1]])
    # axs[0].set_ylabel(r"space $x$")
    # axs[0].set_title(r"$\Delta$")
    axs[0].set_xticklabels(["0s", r"$350s$", r"$700s$"])
    axs[0].grid()
    axs[0].legend(loc='lower right')

    axs[1].plot(t, np.zeros_like(t), color="green", linestyle='-')
    axs[1].set_xticks([0, t[-1] / 2, t[-1]])
    # axs[1].set_ylabel(r"space $x$")
    # axs[1].set_title(r"$\Delta$")
    axs[1].set_xticklabels(["0s", r"$350s$", r"$700s$"])
    axs[1].grid()

    axs[2].plot(t, SHIFTS_TEST[1].flatten()[:Nt//2], color="green", linestyle='-', label='actual')
    axs[2].plot(t, shifts_3_pred.flatten()[:Nt//2], color="red", linestyle='--', label='NN')
    axs[2].plot(t, delta_pred_frame_wise[2][:Nt//2], color="blue", linestyle='--', label='interpolated')
    axs[2].set_xticks([0, t[-1] / 2, t[-1]])
    # axs[2].set_ylabel(r"space $x$")
    # axs[2].set_title(r"$\Delta$")
    axs[2].set_xticklabels(["0s", r"$350s$", r"$700s$"])
    axs[2].grid()
    axs[2].legend(loc='lower left')

    fig.supylabel(r"$x$")
    fig.tight_layout()
    save_fig(filepath=impath + "shifts_sPOD_predicted", figure=fig)


def plot_timeamplitudesPredPOD(POD_frame_amplitudes_predicted, TA_POD_TEST, x, t, nmodes):
    Nx = len(x)
    Nt = len(t)
    fig, axs = plt.subplots(1, nmodes, sharey=True, figsize=(18, 5), num=8)
    plt.subplots_adjust(wspace=0)
    for k, ax in enumerate(axs):
        ax.plot(t, POD_frame_amplitudes_predicted[k, :], color="red", marker=".", label='predicted')
        ax.plot(t, TA_POD_TEST[k, :], color="blue", marker=".", label='actual')
        ax.set_xticks([0, t[-1] / 2, t[-1]])
        ax.set_title(r'${mode}^{(' + str(k) + ')}$')
        ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
        ax.legend(loc='upper right')
        ax.grid()
    fig.supxlabel(r"time $t$")
    fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")
    fig.tight_layout()
    save_fig(filepath=impath + "time_amplitudes_POD_DL_ROM_predicted", figure=fig)


def plot_timeamplitudesPredPOD_few(POD_frame_amplitudes_predicted, TA_POD_TEST, x, t):
    Nx = len(x)
    Nt = len(t)
    t = t[:Nt//2]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), num=23)
    plt.subplots_adjust(wspace=1)
    for k, ax in enumerate(axs):
        ax.plot(t, TA_POD_TEST[k, :Nt//2], color="green", linestyle='-', label='actual')
        ax.plot(t, POD_frame_amplitudes_predicted[k, :Nt//2], color="red", linestyle='--', label='NN')
        ax.set_xticks([0, t[-1] / 2, t[-1]])
        # ax.set_title(r'${mode}^{(' + str(k + 1) + ')}$')
        ax.set_xticklabels(["0s", r"$350s$", r"$700s$"])
        ax.legend(loc='lower center')
        ax.grid()
    # fig.supxlabel(r"time $t$")
    fig.supylabel(r"$a_i^{k}(t,\mu)$")
    fig.tight_layout()
    save_fig(filepath=impath + "time_amplitudes_POD_DL_ROM_predicted_few", figure=fig)


def plot_shiftsPred(shifts_1_pred, shifts_3_pred, SHIFTS_TEST, delta_pred_frame_wise, x, t):

    Nx = len(x)
    Nt = len(t)
    # Frame 1
    fig, ax = plt.subplots(1, 1, figsize=(15, 6), num=9)
    plt.subplots_adjust(wspace=0)
    ax.plot(t, SHIFTS_TEST[0].flatten(), color="red", marker=".", label='actual')
    ax.plot(t, shifts_1_pred.flatten(), color="blue", marker=".", label='predicted')
    ax.plot(t, delta_pred_frame_wise[0], color="yellow", marker=".", label='interpolated')
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"space $x$")
    ax.set_title(r'${frame}^{(' + str(1) + ')}$')
    ax.grid()
    ax.legend(loc='upper right')
    fig.tight_layout()
    save_fig(filepath=impath + "shifts_frame_1" + "predicted", figure=fig)

    # Frame 3
    fig, ax = plt.subplots(1, 1, figsize=(15, 6), num=10)
    plt.subplots_adjust(wspace=0)
    ax.plot(t, SHIFTS_TEST[1].flatten(), color="red", marker=".", label='actual')
    ax.plot(t, shifts_3_pred.flatten(), color="blue", marker=".", label='predicted')
    ax.plot(t, delta_pred_frame_wise[2], color="yellow", marker=".", label='interpolated')
    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"space $x$")
    ax.set_title(r'${frame}^{(' + str(3) + ')}$')
    ax.grid()
    ax.legend(loc='upper right')
    fig.tight_layout()
    save_fig(filepath=impath + "shifts_frame_3" + "predicted", figure=fig)


def plot_recons_snapshot(q_test, Q_recon_interp, Q_recon_sPOD, Q_recon_POD, x, t):

    Nx = len(x)
    Nt = len(t)

    [Xgrid, Tgrid] = meshgrid(x, t)
    Xgrid = Xgrid.T
    Tgrid = Tgrid.T

    qmin = np.min(q_test)
    qmax = np.max(q_test)
    fig, axs = plt.subplots(1, 4, num=11, sharey=True, figsize=(15, 6))
    # Original
    axs[0].pcolormesh(Xgrid, Tgrid, q_test, vmin=qmin, vmax=qmax)
    axs[0].set_title(r"$Q^" + "{original}$")
    axs[0].set_yticks([], [])
    axs[0].set_xticks([], [])
    # Interpolated
    axs[1].pcolormesh(Xgrid, Tgrid, Q_recon_interp, vmin=qmin, vmax=qmax)
    axs[1].set_title(r"$Q^" + "{interpolated}$")
    axs[1].set_yticks([], [])
    axs[1].set_xticks([], [])
    # sPOD NN predicted
    axs[2].pcolormesh(Xgrid, Tgrid, Q_recon_sPOD, vmin=qmin, vmax=qmax)
    axs[2].set_title(r"$Q^" + "{sPOD}_{NN}$")
    axs[2].set_yticks([], [])
    axs[2].set_xticks([], [])
    # POD NN predicted
    axs[3].pcolormesh(Xgrid, Tgrid, Q_recon_POD, vmin=qmin, vmax=qmax)
    axs[3].set_title(r"$Q^" + "{POD}_{NN}$")
    axs[3].set_yticks([], [])
    axs[3].set_xticks([], [])
    plt.tight_layout()

    save_fig(filepath=impath + "Snapshot_Comparison", figure=fig)


def plot_recons_snapshot_cross_section(q_test, Q_recon_interp, Q_recon_sPOD, Q_recon_POD, x, t):

    Nx = len(x)
    Nt = len(t)

    [Xgrid, Tgrid] = meshgrid(x, t)
    Xgrid = Xgrid.T
    Tgrid = Tgrid.T

    qmin = np.min(q_test)
    qmax = np.max(q_test)
    fig, axs = plt.subplots(1, 1, num=30, sharey=True, figsize=(4, 6))
    # Original
    axs.pcolormesh(Xgrid, Tgrid, q_test, vmin=qmin, vmax=qmax)
    axs.axhline(y=t[Nt//30], linestyle='--', color='r', label=r"$Nt=200$")
    axs.axhline(y=t[3*Nt//4], linestyle='--', color='g', label=r"$Nt=4500$")
    axs.set_title("Q")
    axs.set_yticks([], [])
    axs.set_xticks([], [])
    axs.legend(loc='upper center', fontsize=12)
    plt.tight_layout()
    save_fig(filepath=impath + "Snapshot_original", figure=fig)

    x_trim = x[600:2400]
    fig, axs = plt.subplots(2, 1, figsize=(7, 12), num=31)
    plt.subplots_adjust(wspace=1)
    axs[0].plot(x_trim, q_test[600:2400, 3*Nt//4], color="green", linestyle='-', label='actual')
    axs[0].plot(x_trim, Q_recon_interp[600:2400, 3*Nt//4], color="blue", linestyle='--', label='interpolated')
    axs[0].plot(x_trim, Q_recon_sPOD[600:2400, 3*Nt//4], color="red", linestyle='--', label='sPOD DL')
    axs[0].plot(x_trim, Q_recon_POD[600:2400, 3*Nt//4], color="black", linestyle='--', label='POD DL')
    axin2 = axs[0].inset_axes([0.35, 0.2, 0.30, 0.30])
    axin2.plot(x[810:900], q_test[810:900, 3*Nt//4], color="green", linestyle='-')
    axin2.plot(x[810:900], Q_recon_interp[810:900, 3*Nt//4], color="blue", linestyle='--')
    axin2.plot(x[810:900], Q_recon_sPOD[810:900, 3*Nt//4], color="red", linestyle='--')
    axin2.plot(x[810:900], Q_recon_POD[810:900, 3*Nt//4], color="black", linestyle='--')
    axin2.set_xlim(270, 300)
    axin2.set_ylim(-60, 60)
    axin2.set_xticks([], [])
    axin2.set_yticks([], [])
    axs[0].indicate_inset_zoom(axin2)
    # axs[0].set_yticks([], [])
    # axs[0].set_xticks([], [])
    axs[0].grid()
    axs[0].legend(loc='upper center', fontsize=20)

    axs[1].plot(x_trim, q_test[600:2400, Nt//30], color="green", linestyle='-', label='actual')
    axs[1].plot(x_trim, Q_recon_interp[600:2400, Nt//30], color="blue", linestyle='--', label='interpolated')
    axs[1].plot(x_trim, Q_recon_sPOD[600:2400, Nt//30], color="red", linestyle='--', label='sPOD DL')
    axs[1].plot(x_trim, Q_recon_POD[600:2400, Nt//30], color="black", linestyle='--', label='POD DL')

    axin = axs[1].inset_axes([0.6, 0.2, 0.3, 0.3])
    axin.plot(x[1560:1650], q_test[1560:1650, Nt//30], color="green", linestyle='-')
    axin.plot(x[1560:1650], Q_recon_interp[1560:1650, Nt//30], color="blue", linestyle='--')
    axin.plot(x[1560:1650], Q_recon_sPOD[1560:1650, Nt//30], color="red", linestyle='--')
    axin.plot(x[1560:1650], Q_recon_POD[1560:1650, Nt//30], color="black", linestyle='--')
    axin.set_xlim(520, 550)
    axin.set_ylim(-50, 50)
    axin.set_xticks([], [])
    axin.set_yticks([], [])
    axs[1].indicate_inset_zoom(axin)
    # axs[1].set_yticks([], [])
    # axs[1].set_xticks([], [])
    axs[1].grid()
    axs[1].legend(loc='upper left', fontsize=20)


    fig.supylabel(r"$T$", fontsize=20)
    fig.supxlabel(r"$x$", fontsize=20)
    fig.tight_layout()
    save_fig(filepath=impath + "T_x_cross_section", figure=fig)
