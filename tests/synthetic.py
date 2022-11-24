import numpy as np
from scipy.special import eval_hermite
import os
from Helper import *

impath = "./plots/images_combustionWave/"
os.makedirs(impath, exist_ok=True)


class CombustionWave:
    def __init__(self, spod_iter):
        self.Nx = 500  # number of grid points in x
        self.Nt = 500  # numer of time intervals

        self.NumFrames = 2

        self.T = 1  # total time
        self.L = 1  # total domain size
        self.nmodes = 5  # reduction of singular values
        self.D = self.nmodes

        self.x = np.arange(-self.Nx // 2, self.Nx // 2) / self.Nx * self.L
        self.t = np.arange(0, self.Nt) / self.Nt * self.T

        [self.Xgrid, self.Tgrid] = np.meshgrid(self.x, self.t)
        self.Xgrid = self.Xgrid.T
        self.Tgrid = self.Tgrid.T

        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]
        c = 1

        print("#############################################")
        print("Synthetic data checks....")
        print("Check 1...")
        ##########################################
        # %% Create training data
        mu_vecs_train = np.asarray([0.1, 0.2, 0.3])
        Nsamples_train = np.size(mu_vecs_train)
        q_train, q1_train, q2_train, shifts_train, params_train, trafos_train = self.create_data(mu_vecs_train)

        U1, S1, VT1 = np.linalg.svd(q1_train, full_matrices=False)
        U2, S2, VT2 = np.linalg.svd(q2_train, full_matrices=False)
        err1 = np.sqrt(
            np.mean(
                np.linalg.norm(q1_train - U1[:, :self.D] @ (np.diag(S1[:self.D]) @ VT1[:self.D, :]), 2, axis=1) ** 2)) \
               / np.sqrt(np.mean(np.linalg.norm(q1_train, 2, axis=0) ** 2))
        err2 = np.sqrt(
            np.mean(
                np.linalg.norm(q2_train - U2[:, :self.D] @ (np.diag(S2[:self.D]) @ VT2[:self.D, :]), 2, axis=1) ** 2)) \
               / np.sqrt(np.mean(np.linalg.norm(q2_train, 2, axis=0) ** 2))
        print("Error for frame 1 SVD recons. with {} number of modes is {}".format(self.D, err1))
        print("Error for frame 2 SVD recons. with {} number of modes is {}".format(self.D, err2))

        ##########################################
        # %% Create testing data
        mu_vecs_test = np.asarray([0.15])
        Nsamples_test = np.size(mu_vecs_test)
        self.q_test, self.q1_test, self.q2_test, self.shifts_test, self.params_test, self.trafos_test = \
            self.create_data(mu_vecs_test)

        ##########################################
        # %% Calculate the transformation interpolation error
        interp_err = give_interpolation_error(q_train, trafos_train[0])
        print("Check 2...")
        print("Transformation interpolation error =  %4.4e " % interp_err)

        # Calculate the time amplitudes
        # qmat = np.reshape(q_train, [-1, Nsamples_train * self.Nt])
        # [N, M] = np.shape(qmat)
        # mu0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.0001
        # lambd0 = 1 / np.sqrt(np.maximum(M, N)) * 10
        # ret = shifted_rPCA(q_train, trafos_train, nmodes_max=np.max(self.nmodes) + 10, eps=1e-16, Niter=spod_iter,
        #                    use_rSVD=True,
        #                    mu=mu0, lambd=lambd0)
        ret = shifted_POD(q_train, trafos_train, self.nmodes, eps=1e-16, Niter=spod_iter, use_rSVD=True)
        sPOD_frames, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist

        ###########################################
        # %% relative offline error for training data (srPCA error)
        q1_spod_frame = sPOD_frames[0].build_field()
        q2_spod_frame = sPOD_frames[1].build_field()
        err_full = np.sqrt(np.mean(np.linalg.norm(q_train - qtilde, 2, axis=1) ** 2)) / \
                   np.sqrt(np.mean(np.linalg.norm(q_train, 2, axis=1) ** 2))
        print("Check 3...")
        print("Error for full sPOD recons. is {}".format(err_full))

        ###########################################
        # %% Calculate the time amplitudes for training data
        self.U_list = []
        frame_amplitude_list = []
        frame_amplitude_list_training = []
        shifts_list = []
        cnt = 0
        for frame in sPOD_frames:
            VT = frame.modal_system["VT"]
            S = frame.modal_system["sigma"]
            VT = np.diag(S) @ VT
            Nmodes = frame.Nmodes
            amplitudes = [np.reshape(VT[n, :], [Nsamples_train, self.Nt]).T for n in range(Nmodes)]
            shifts = np.reshape(shifts_train[cnt], [Nsamples_train, self.Nt]).T
            frame_amplitude_list.append(amplitudes)
            frame_amplitude_list_training.append(VT)
            self.U_list.append(frame.modal_system["U"])
            shifts_list.append(shifts)
            cnt = cnt + 1

        ###########################################
        # %% Implement the interpolation to find the online prediction
        Nmodes = [self.nmodes, self.nmodes]
        self.qtilde_frame_wise, self.TA_interpolated = my_interpolated_state(Nmodes, self.U_list, frame_amplitude_list,
                                                                             mu_vecs_train,
                                                                             self.Nx, self.Nt,
                                                                             mu_vecs_test, self.trafos_test)
        self.delta_pred_frame_wise = my_delta_interpolate(shifts_list, mu_vecs_train, mu_vecs_test)

        num = np.sqrt(np.mean(np.linalg.norm(self.q_test - self.qtilde_frame_wise, 2, axis=1) ** 2))
        den = np.sqrt(np.mean(np.linalg.norm(self.q_test, 2, axis=1) ** 2))
        num2 = np.sqrt(np.mean(np.linalg.norm(self.shifts_test[0] - self.delta_pred_frame_wise[0], 2, axis=0) ** 2))
        den2 = np.sqrt(np.mean(np.linalg.norm(self.shifts_test[0], 2, axis=0) ** 2))
        num3 = np.sqrt(np.mean(np.linalg.norm(self.shifts_test[1] - self.delta_pred_frame_wise[1], 2, axis=0) ** 2))
        den3 = np.sqrt(np.mean(np.linalg.norm(self.shifts_test[1], 2, axis=0) ** 2))
        print("Check 4...")
        print(
            "Relative full recon. error for online prediction(interpolation frame wise) for num modes {} is {}".format(
                self.nmodes, num / den))
        print(
            "Relative shift error for online prediction(interpolation) for frame {} is {}".format(
                1, num2 / den2))
        print(
            "Relative shift error for online prediction(interpolation) for frame {} is {}".format(
                2, num3 / den3))
        # plot the frames
        self.plot_sPODframes(q_train, qtilde, q1_spod_frame, q2_spod_frame)

        ###########################################
        # %% Generate data for the POD-DL-ROM for comparison
        U, S, VT = np.linalg.svd(np.squeeze(q_train), full_matrices=False)
        self.U_POD_TRAIN = U[:, :sum(Nmodes) + 2]
        self.TA_POD_TRAIN = np.diag(S[:sum(Nmodes) + 2]) @ VT[:sum(Nmodes) + 2, :]
        self.TA_POD_TEST = self.U_POD_TRAIN.transpose() @ self.q_test

        ###########################################
        # %% data for the NN
        amplitudes_train = np.concatenate(frame_amplitude_list_training, axis=0)
        time_amplitudes_1_test = self.U_list[0][:, :self.D].transpose() @ self.q1_test
        time_amplitudes_2_test = self.U_list[1][:, :self.D].transpose() @ self.q2_test
        amplitudes_test = np.concatenate((time_amplitudes_1_test, time_amplitudes_2_test), axis=0)

        self.TA_TRAIN = amplitudes_train
        self.SHIFTS_TRAIN = [shifts_train[0], shifts_train[1]]
        self.PARAMS_TRAIN = params_train
        self.TA_TEST = amplitudes_test
        self.SHIFTS_TEST = [self.shifts_test[0], self.shifts_test[1]]
        self.PARAMS_TEST = self.params_test

        ###########################################
        # %% Plot all the variables required
        self.plot_trainingframes(q_train, q1_train, q2_train, Nsamples_train)
        self.plot_trainingshift(shifts_train, mu_vecs_train)
        self.plot_timeamplitudes(frame_amplitude_list_training)

    def create_data(self, mu_vecs):

        # gauss hermite polynomials of order n
        psi = lambda n, x: np.exp(-x ** 2 / 2) * eval_hermite(n, x)
        Nsamples = np.size(mu_vecs)
        w = 0.015 * self.L
        Nt = np.size(self.Tgrid, 1)

        qs1 = []
        qs2 = []
        for k in range(Nsamples):  # loop over all possible mu vectors
            q1 = np.zeros_like(self.Xgrid)
            q2 = np.zeros_like(self.Xgrid)
            for n in range(self.D):  # loop over all components of the vector
                q1 += (1 + np.exp(-2 * n * self.Tgrid)) * mu_vecs[k] * np.cos(
                    - 2 * np.pi * self.Tgrid / self.T * (n + 1)) * psi(n, (
                        self.Xgrid + 0.1 * self.L) / w)
                q2 += (1 + np.exp(-2 * n * self.Tgrid)) * mu_vecs[k] * np.cos(
                    - 2 * np.pi * self.Tgrid / self.T * (n + 1)) * psi(n, (
                        self.Xgrid - 0.1 * self.L) / w)

            qs1.append(q1)
            qs2.append(-q2)

        q1 = np.concatenate(qs1, axis=1)
        q2 = np.concatenate(qs2, axis=1)
        q_frames = [q1, q2]

        shifts = [np.asarray([mu_vecs[n] * self.t]) for n in range(Nsamples)]

        shifts = [np.concatenate(shifts, axis=1), -np.concatenate(shifts, axis=1)]
        data_shape = [self.Nx, 1, 1, Nt * Nsamples]
        trafos = [
            transforms(data_shape, [self.L], shifts=np.squeeze(shifts[0]), dx=[self.dx], use_scipy_transform=False,
                       interp_order=5),
            transforms(data_shape, [self.L], shifts=np.squeeze(shifts[1]), dx=[self.dx], use_scipy_transform=False,
                       interp_order=5)]

        q = 0
        for trafo, qf in zip(trafos, q_frames):
            q += trafo.apply(qf)

        # Parameter matrix
        p = [np.squeeze(np.asarray([[self.t], [np.ones_like(self.t) * mu]])) for mu in mu_vecs]
        p = np.concatenate(p, axis=1)

        return q, q1, q2, shifts, p, trafos

    def onlineErroranalysis(self, frame_amplitude_predicted, shifts_predicted, POD_frame_amplitudes_predicted):
        print("#############################################")
        print('Online Error checks')
        ###########################################
        # %% Online error with respect to testing data
        time_amplitudes_1_pred = frame_amplitude_predicted[:self.D, :]
        time_amplitudes_2_pred = frame_amplitude_predicted[self.D:2 * self.D, :]
        shifts_1_pred = shifts_predicted[0, :]
        shifts_2_pred = shifts_predicted[1, :]

        # Time amplitudes error
        time_amplitudes_1_test = self.TA_TEST[:self.D, :]
        time_amplitudes_2_test = self.TA_TEST[self.D:2 * self.D, :]
        num1 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_1_test - time_amplitudes_1_pred, 2, axis=1) ** 2))
        den1 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_1_test, 2, axis=1) ** 2))
        num2 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_2_test - time_amplitudes_2_pred, 2, axis=1) ** 2))
        den2 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_2_test, 2, axis=1) ** 2))
        num3 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_1_test - self.TA_interpolated[0], 2, axis=1) ** 2))
        den3 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_1_test, 2, axis=1) ** 2))
        num4 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_2_test - self.TA_interpolated[1], 2, axis=1) ** 2))
        den4 = np.sqrt(np.mean(np.linalg.norm(time_amplitudes_2_test, 2, axis=1) ** 2))
        num5 = np.sqrt(np.mean(np.linalg.norm(self.TA_POD_TEST - POD_frame_amplitudes_predicted, 2, axis=1) ** 2))
        den5 = np.sqrt(np.mean(np.linalg.norm(self.TA_POD_TEST, 2, axis=1) ** 2))
        print('Check 1...')
        print("Relative time amplitude error indicator for frame: 1 is {}".format(num1 / den1))
        print("Relative time amplitude error indicator for frame: 2 is {}".format(num2 / den2))
        print("Relative time amplitude error indicator (interpolation) for frame: 1 is {}".format(num3 / den3))
        print("Relative time amplitude error indicator (interpolation) for frame: 2 is {}".format(num4 / den4))
        print("Relative time amplitude error indicator for POD-DL-ROM is {}".format(num5 / den5))

        # Shifts error
        num1 = np.sqrt(np.mean(np.linalg.norm(self.SHIFTS_TEST[0] - shifts_1_pred.flatten(), 2, axis=0) ** 2))
        den1 = np.sqrt(np.mean(np.linalg.norm(self.SHIFTS_TEST[0], 2, axis=0) ** 2))
        num2 = np.sqrt(np.mean(np.linalg.norm(self.SHIFTS_TEST[1] - shifts_2_pred.flatten(), 2, axis=0) ** 2))
        den2 = np.sqrt(np.mean(np.linalg.norm(self.SHIFTS_TEST[1], 2, axis=0) ** 2))
        print('Check 2...')
        print("Relative error indicator for shift: 1 is {}".format(num1 / den1))
        print("Relative error indicator for shift: 2 is {}".format(num2 / den2))

        # Frame wise error
        q1_pred = self.U_list[0][:, :self.D] @ time_amplitudes_1_pred
        q2_pred = self.U_list[1][:, :self.D] @ time_amplitudes_2_pred
        num1 = np.sqrt(np.mean(np.linalg.norm(self.q1_test - q1_pred, 2, axis=1) ** 2))
        den1 = np.sqrt(np.mean(np.linalg.norm(self.q1_test, 2, axis=1) ** 2))
        num2 = np.sqrt(np.mean(np.linalg.norm(self.q2_test - q2_pred, 2, axis=1) ** 2))
        den2 = np.sqrt(np.mean(np.linalg.norm(self.q2_test, 2, axis=1) ** 2))
        print('Check 3...')
        print("Relative frame snapshot reconstruction error indicator for frame: 1 is {}".format(num1 / den1))
        print("Relative frame snapshot reconstruction error indicator for frame: 2 is {}".format(num2 / den2))

        # Total reconstructed error
        use_original_shift = True
        Q_recon = 0
        NumFrames = 2
        Q_pred = [q1_pred, q2_pred]
        if use_original_shift:
            for frame in range(NumFrames):
                Q_recon += self.trafos_test[frame].apply(Q_pred[frame])
        else:
            trafos = [
                transforms([self.Nx, 1, 1, self.Nt], [self.L], shifts=np.squeeze(shifts_1_pred), dx=[self.dx],
                           use_scipy_transform=False,
                           interp_order=5),
                transforms([self.Nx, 1, 1, self.Nt], [self.L], shifts=np.squeeze(shifts_2_pred), dx=[self.dx],
                           use_scipy_transform=False,
                           interp_order=5)]
            for frame in NumFrames:
                Q_recon += trafos[frame].apply(Q_pred[frame])

        num1 = np.sqrt(np.mean(np.linalg.norm(self.q_test - Q_recon, 2, axis=1) ** 2))
        den1 = np.sqrt(np.mean(np.linalg.norm(self.q_test, 2, axis=1) ** 2))

        POD_DL_ROM_recon = self.U_POD_TRAIN @ POD_frame_amplitudes_predicted
        num2 = np.sqrt(np.mean(np.linalg.norm(self.q_test - POD_DL_ROM_recon, 2, axis=1) ** 2))
        den2 = np.sqrt(np.mean(np.linalg.norm(self.q_test, 2, axis=1) ** 2))
        print('Check 4...')
        print("Relative reconstruction error indicator for full snapshot(sPOD-DL-ROM) is {}".format(num1 / den1))
        print("Relative reconstruction error indicator for full snapshot(POD-DL-ROM) is {}".format(num2 / den2))

        # Plot the online prediction data
        self.plot_timeamplitudesPred(time_amplitudes_1_pred, time_amplitudes_1_test, time_amplitudes_2_pred,
                                     time_amplitudes_2_test)
        self.plot_timeamplitudesPredPOD(POD_frame_amplitudes_predicted, nmodes=5)
        self.plot_shiftsPred(shifts_1_pred, shifts_2_pred)
        self.plot_recons_snapshot(Q_recon, POD_DL_ROM_recon)

    def plot_trainingframes(self, q_train, q1_train, q2_train, Nsamples_train):
        Nx = self.Nx
        Nt = self.Nt
        fig, axs = plt.subplots(3, Nsamples_train, sharey=True, sharex=True, figsize=(10, 6), num=1)
        plt.subplots_adjust(wspace=0)
        qmin = np.min(q_train)
        qmax = np.max(q_train)
        for k in range(0, Nsamples_train):
            kw = k
            axs[0, kw].pcolormesh(q_train[:, Nt * k:Nt * (k + 1)], vmin=qmin, vmax=qmax, cmap=cm)
            axs[0, kw].set_title(r'${\mu}^{(' + str(k) + ')}$')
            axs[0, kw].set_yticks([0, Nx // 2, Nx])
            axs[0, kw].set_xticks([0, Nt // 2, Nt])
            axs[0, kw].set_yticklabels([r"$-L/2$", 0, r"$L/2$"])
            axs[1, kw].pcolormesh(q1_train[:, Nt * k:Nt * (k + 1)], vmin=qmin, vmax=qmax, cmap=cm)
            im = axs[2, kw].pcolormesh(q2_train[:, Nt * k:Nt * (k + 1)], vmin=qmin, vmax=qmax, cmap=cm)

        axs[0, 0].set_ylabel(r"$q$")
        axs[1, 0].set_ylabel(r"$q_1$")
        axs[2, 0].set_ylabel(r"$q_2$")
        axs[0, 0].set_xticklabels(["", r"$T/2$", r"$T$"])
        fig.supxlabel(r"time $t$")
        fig.supylabel(r"space $x$")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.83, 0.25, 0.01, 0.5])
        fig.colorbar(im, cax=cbar_ax)
        fig.savefig(impath + "Combustionwaves_" + "training" + '.png', dpi=600, transparent=True)

    def plot_trainingshift(self, shifts_train, mu_vecs_train):
        Nt = len(self.t)
        fig, axs = plt.subplots(1, len(mu_vecs_train), figsize=(15, 6), num=2)
        plt.subplots_adjust(wspace=0)
        for k, ax in enumerate(axs):
            ax.plot(self.t, shifts_train[0][:, k * Nt:(k + 1) * Nt].flatten(), color="red", marker=".", label='frame 1')
            ax.plot(self.t, shifts_train[1][:, k * Nt:(k + 1) * Nt].flatten(), color="blue", marker=".",
                    label='frame 2')
            ax.set_title(r'${\mu}^{(' + str(k) + ')}$')
            ax.grid()
            ax.legend(loc='center right')
        fig.supxlabel(r"time $t$")
        fig.supylabel(r"space $x$")
        fig.tight_layout()
        save_fig(filepath=impath + "shifts_" + "training", figure=fig)

    def plot_sPODframes(self, q_train, qtilde, q1_spod_frame, q2_spod_frame):
        qmin = np.min(q_train)
        qmax = np.max(q_train)
        fig, axs = plt.subplots(1, 3, num=3, sharey=True, figsize=(10, 3))
        # 1. frame
        k_frame = 0
        axs[0].pcolormesh(q1_spod_frame[:, :self.Nt], cmap=cm, vmin=qmin, vmax=qmax)
        axs[0].set_ylabel(r'$i=1,\dots,M$')
        axs[0].set_xlabel(r'$j=1,\dots,ND$')
        axs[0].set_title(r"$Q^" + str(k_frame + 1) + "_{i,j}$")
        # 2. frame
        k_frame = 1
        axs[1].pcolormesh(q2_spod_frame[:, :self.Nt], cmap=cm, vmin=qmin, vmax=qmax)
        axs[1].set_xlabel(r'$j=1,\dots,ND$')
        axs[1].set_title(r"$Q^" + str(k_frame + 1) + "_{i,j}$")
        # Reconstruction
        axs[2].pcolormesh(qtilde[:, :self.Nt], cmap=cm, vmin=qmin, vmax=qmax)
        axs[2].set_xlabel(r'$j=1,\dots,ND$')
        axs[2].set_title(r"$\tilde{Q}" + "_{i,j}$")
        plt.tight_layout()

        save_fig(filepath=impath + "Combustionwaves_spod_frames", figure=fig)

    def plot_timeamplitudes(self, frame_amplitude_list_training):
        fig, axs = plt.subplots(1, self.D, sharey=True, figsize=(18, 5), num=6)
        plt.subplots_adjust(wspace=0)
        for k, ax in enumerate(axs):
            ax.plot(self.t, frame_amplitude_list_training[0][k, 0:self.Nt], color="red", marker=".", label='frame 1')
            ax.plot(self.t, frame_amplitude_list_training[1][k, 0:self.Nt], color="blue", marker=".", label='frame 2')
            ax.set_xticks([0, self.t[-1] / 2, self.t[-1]])
            ax.set_title(r'${mode}^{(' + str(k) + ')}$')
            ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
            ax.grid()
            ax.legend(loc='upper right')
        fig.supxlabel(r"time $t$")
        fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")
        fig.tight_layout()
        save_fig(filepath=impath + "time_amplitudes_" + "training", figure=fig)

    def plot_timeamplitudesPred(self, time_amplitudes_1_pred, time_amplitudes_1_test, time_amplitudes_2_pred,
                                time_amplitudes_2_test):
        # Frame 1
        fig, axs = plt.subplots(1, self.D, sharey=True, figsize=(18, 5), num=7)
        plt.subplots_adjust(wspace=0)
        for k, ax in enumerate(axs):
            ax.plot(self.t, time_amplitudes_1_pred[k, :], color="red", marker=".", label='predicted')
            ax.plot(self.t, time_amplitudes_1_test[k, :], color="blue", marker=".", label='actual')
            ax.plot(self.t, self.TA_interpolated[0][k, :], color="yellow", marker=".", label='interpolated')
            ax.set_xticks([0, self.t[-1] / 2, self.t[-1]])
            ax.set_title(r'${mode}^{(' + str(k) + ')}$')
            ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
            ax.legend(loc='upper right')
            ax.grid()
        fig.supxlabel(r"time $t$")
        fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")
        fig.tight_layout()
        save_fig(filepath=impath + "time_amplitudes_frame_1_" + "predicted", figure=fig)

        # Frame 2
        fig, axs = plt.subplots(1, self.D, sharey=True, figsize=(18, 5), num=8)
        plt.subplots_adjust(wspace=0)
        for k, ax in enumerate(axs):
            ax.plot(self.t, time_amplitudes_2_pred[k, :], color="red", marker=".", label='predicted')
            ax.plot(self.t, time_amplitudes_2_test[k, :], color="blue", marker=".", label='actual')
            ax.plot(self.t, self.TA_interpolated[1][k, :], color="yellow", marker=".", label='interpolated')
            ax.set_xticks([0, self.t[-1] / 2, self.t[-1]])
            ax.set_title(r'${mode}^{(' + str(k) + ')}$')
            ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
            ax.legend(loc='upper right')
            ax.grid()
        fig.supxlabel(r"time $t$")
        fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")
        fig.tight_layout()
        save_fig(filepath=impath + "time_amplitudes_frame_2_" + "predicted", figure=fig)

    def plot_timeamplitudesPredPOD(self, POD_frame_amplitudes_predicted, nmodes):
        fig, axs = plt.subplots(1, nmodes, sharey=True, figsize=(18, 5), num=9)
        plt.subplots_adjust(wspace=0)
        for k, ax in enumerate(axs):
            ax.plot(self.t, POD_frame_amplitudes_predicted[k, :], color="red", marker=".", label='predicted')
            ax.plot(self.t, self.TA_POD_TEST[k, :], color="blue", marker=".", label='actual')
            ax.set_xticks([0, self.t[-1] / 2, self.t[-1]])
            ax.set_title(r'${mode}^{(' + str(k) + ')}$')
            ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
            ax.legend(loc='upper right')
            ax.grid()
        fig.supxlabel(r"time $t$")
        fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")
        fig.tight_layout()
        save_fig(filepath=impath + "time_amplitudes_POD_DL_ROM_" + "predicted", figure=fig)

    def plot_shiftsPred(self, shifts_1_pred, shifts_2_pred):

        # Frame 1
        fig, ax = plt.subplots(1, 1, figsize=(15, 6), num=10)
        plt.subplots_adjust(wspace=0)
        ax.plot(self.t, self.SHIFTS_TEST[0].flatten(), color="red", marker=".", label='actual')
        ax.plot(self.t, shifts_1_pred.flatten(), color="blue", marker=".", label='predicted')
        ax.plot(self.t, self.delta_pred_frame_wise[0], color="yellow", marker=".", label='interpolated')
        ax.set_xlabel('time', fontsize=14)
        ax.set_ylabel('space', fontsize=14)
        ax.set_title(r'${frame}^{(' + str(1) + ')}$')
        ax.grid()
        ax.legend(loc='center right')
        fig.tight_layout()
        save_fig(filepath=impath + "shifts_frame_1" + "predicted", figure=fig)

        # Frame 2
        fig, ax = plt.subplots(1, 1, figsize=(15, 6), num=11)
        plt.subplots_adjust(wspace=0)
        ax.plot(self.t, self.SHIFTS_TEST[1].flatten(), color="red", marker=".", label='actual')
        ax.plot(self.t, shifts_2_pred.flatten(), color="blue", marker=".", label='predicted')
        ax.plot(self.t, self.delta_pred_frame_wise[1], color="yellow", marker=".", label='interpolated')
        ax.set_xlabel('time', fontsize=14)
        ax.set_ylabel('space', fontsize=14)
        ax.set_title(r'${frame}^{(' + str(2) + ')}$')
        ax.grid()
        ax.legend(loc='center right')
        fig.tight_layout()
        save_fig(filepath=impath + "shifts_frame_2" + "predicted", figure=fig)

    def plot_recons_snapshot(self, Q_recon, POD_DL_ROM_recon):
        qmin = np.min(self.q_test)
        qmax = np.max(self.q_test)
        fig, axs = plt.subplots(1, 4, num=12, sharey=True, figsize=(10, 3))
        # Original
        axs[0].pcolormesh(self.q_test, cmap=cm, vmin=qmin, vmax=qmax)
        axs[0].set_ylabel(r'$i=1,\dots,M$')
        axs[0].set_xlabel(r'$j=1,\dots,ND$')
        axs[0].set_title(r"$Q^" + "{original}" + "_{i,j}$")
        # Interpolated
        axs[1].pcolormesh(self.qtilde_frame_wise, cmap=cm, vmin=qmin, vmax=qmax)
        axs[1].set_xlabel(r'$j=1,\dots,ND$')
        axs[1].set_title(r"$Q^" + "{interpolated}" + "_{i,j}$")
        # sPOD NN predicted
        axs[2].pcolormesh(Q_recon, cmap=cm, vmin=qmin, vmax=qmax)
        axs[2].set_xlabel(r'$j=1,\dots,ND$')
        axs[2].set_title(r"$Q^" + "{sPODNN}" + "_{i,j}$")
        # POD NN predicted
        axs[3].pcolormesh(POD_DL_ROM_recon, cmap=cm, vmin=qmin, vmax=qmax)
        axs[3].set_xlabel(r'$j=1,\dots,ND$')
        axs[3].set_title(r"$Q^" + "{PODNN}" + "_{i,j}$")
        plt.tight_layout()

        save_fig(filepath=impath + "Snapshot_Comparison", figure=fig)