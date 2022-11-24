import numpy as np
from scipy.special import eval_hermite
import os
import sys
from Helper import *

impath = "./plots/images_combustionWaveTA/"
os.makedirs(impath, exist_ok=True)


class CombustionWaveTA:
    def __init__(self):
        self.Nx = 500
        self.Nt = 500  # numer of time intervals

        self.NumFrames = 2

        self.L = 1  # total domain size
        self.nmodes = 5  # reduction of singular values
        self.D = self.nmodes
        self.t = np.linspace(0, np.pi, self.Nt)
        self.x = np.arange(-self.Nx // 2, self.Nx // 2) / self.Nx * self.L
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]
        c = 1
        [self.Xgrid, self.Tgrid] = meshgrid(self.x, self.t)
        self.Xgrid = self.Xgrid.T
        self.Tgrid = self.Tgrid.T

        ##########################################
        # %% Create training data
        mu_vecs_train = np.asarray([4, 5, 6])
        Nsamples_train = np.size(mu_vecs_train)
        amplitudes_train, shifts_train, params_train = self.create_data(mu_vecs_train)

        ##########################################
        # %% Create testing data
        mu_vecs_test = np.asarray([5.5])
        Nsamples_test = np.size(mu_vecs_test)
        self.amplitudes_test, self.shifts_test, self.params_test = self.create_data(mu_vecs_test)

        ###########################################
        # %% Implement the interpolation to find the online prediction
        frame_amplitude_list = []
        shifts_list = []
        for frame in range(self.NumFrames):
            VT = amplitudes_train[frame * self.nmodes:(frame + 1) * self.nmodes, :]
            Nmodes = self.nmodes
            amplitudes = [np.reshape(VT[n, :], [Nsamples_train, self.Nt]).T for n in range(Nmodes)]
            shifts = np.reshape(shifts_train[frame], [Nsamples_train, self.Nt]).T
            frame_amplitude_list.append(amplitudes)
            shifts_list.append(shifts)

        Nmodes = [self.nmodes, self.nmodes]
        self.ta_pred_frame_wise = my_interpolated_state_onlyTA(Nmodes, frame_amplitude_list, mu_vecs_train,
                                                               mu_vecs_test)
        self.delta_pred_frame_wise = my_delta_interpolate(shifts_list, mu_vecs_train, mu_vecs_test)

        num1 = np.sqrt(
            np.mean(np.linalg.norm(self.amplitudes_test[:self.nmodes, :] - self.ta_pred_frame_wise[0], 2, axis=1) ** 2))
        den1 = np.sqrt(np.mean(np.linalg.norm(self.amplitudes_test[:self.nmodes, :], 2, axis=1) ** 2))
        num2 = np.sqrt(np.mean(
            np.linalg.norm(self.amplitudes_test[self.nmodes:2 * self.nmodes, :] - self.ta_pred_frame_wise[1], 2,
                           axis=1) ** 2))
        den2 = np.sqrt(np.mean(np.linalg.norm(self.amplitudes_test[self.nmodes:2 * self.nmodes, :], 2, axis=1) ** 2))
        num3 = np.sqrt(np.mean(np.linalg.norm(self.shifts_test[0] - self.delta_pred_frame_wise[0], 2, axis=0) ** 2))
        den3 = np.sqrt(np.mean(np.linalg.norm(self.shifts_test[0], 2, axis=0) ** 2))
        num4 = np.sqrt(np.mean(np.linalg.norm(self.shifts_test[1] - self.delta_pred_frame_wise[1], 2, axis=0) ** 2))
        den4 = np.sqrt(np.mean(np.linalg.norm(self.shifts_test[1], 2, axis=0) ** 2))
        print("Check 1...")
        print(
            "Relative time amplitude error for online prediction(interpolation) for num modes {} is {}".format(
                self.nmodes, num1 / den1))
        print(
            "Relative time amplitude error for online prediction(interpolation) for num modes {} is {}".format(
                self.nmodes, num2 / den2))
        print(
            "Relative shift error for online prediction(interpolation) for num modes {} is {}".format(
                self.nmodes, num3 / den3))
        print(
            "Relative shift error for online prediction(interpolation) for num modes {} is {}".format(
                self.nmodes, num4 / den4))

        ###########################################
        # %% data for the NN
        self.TA_TRAIN = amplitudes_train
        self.SHIFTS_TRAIN = [shifts_train[0], shifts_train[1]]
        self.PARAMS_TRAIN = params_train
        self.TA_TEST = self.amplitudes_test
        self.SHIFTS_TEST = [self.shifts_test[0], self.shifts_test[1]]
        self.PARAMS_TEST = self.params_test

        ###########################################
        # %% Plot all the variables required
        self.plot_trainingshift(shifts_train, mu_vecs_train)
        self.plot_timeamplitudesTraining(amplitudes_train)

    def create_data(self, mu_vecs):
        Nsamples = np.size(mu_vecs)

        amplitudes_1 = np.zeros([self.D, Nsamples * self.Nt])
        amplitudes_2 = np.zeros([self.D, Nsamples * self.Nt])
        for i in range(self.D):
            amplitudes_1[i, :] = np.concatenate(
                np.asarray([(1 + np.sin(mu * self.t) * np.exp(-mu * self.t)) * mu * (i + 1) for mu in mu_vecs]))
            amplitudes_2[i, :] = np.concatenate(
                np.asarray([(1 - np.sin(mu * self.t) * np.exp(-mu * self.t)) * mu * (i + 1) for mu in mu_vecs]))

        amplitudes = np.concatenate((amplitudes_1, amplitudes_2), axis=0)

        shifts = [np.concatenate(np.asarray([(mu * self.t) for mu in mu_vecs])),
                  np.concatenate(np.asarray([(-mu * self.t) for mu in mu_vecs]))]

        p = [np.squeeze(np.asarray([[self.t], [np.ones_like(self.t) * mu]])) for mu in mu_vecs]
        p = np.concatenate(p, axis=1)

        return amplitudes, shifts, p

    def onlineErroranalysis(self, frame_amplitude_predicted, shifts_predicted, POD_frame_amplitudes_predicted=None):
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
        print('Check 1...')
        print("Relative time amplitude error indicator for frame: 1 is {}".format(num1 / den1))
        print("Relative time amplitude error indicator for frame: 2 is {}".format(num2 / den2))

        # Shifts error
        num1 = np.sqrt(np.mean(np.linalg.norm(self.SHIFTS_TEST[0] - shifts_1_pred.flatten(), 2, axis=0) ** 2))
        den1 = np.sqrt(np.mean(np.linalg.norm(self.SHIFTS_TEST[0], 2, axis=0) ** 2))
        num2 = np.sqrt(np.mean(np.linalg.norm(self.SHIFTS_TEST[1] - shifts_2_pred.flatten(), 2, axis=0) ** 2))
        den2 = np.sqrt(np.mean(np.linalg.norm(self.SHIFTS_TEST[1], 2, axis=0) ** 2))
        print('Check 2...')
        print("Relative error indicator for shift: 1 is {}".format(num1 / den1))
        print("Relative error indicator for shift: 2 is {}".format(num2 / den2))

        # Plot the online prediction data
        self.plot_timeamplitudesPred(time_amplitudes_1_pred, time_amplitudes_1_test, time_amplitudes_2_pred,
                                     time_amplitudes_2_test)
        self.plot_shiftsPred(shifts_1_pred, shifts_2_pred)

    def plot_timeamplitudesTraining(self, amplitudes_train):
        fig, axs = plt.subplots(1, self.D, sharey=True, figsize=(18, 5), num=1)
        plt.subplots_adjust(wspace=0)
        for k, ax in enumerate(axs):
            ax.plot(self.t, amplitudes_train[k, :self.Nt], color="red", marker=".", label='frame 1')
            ax.plot(self.t, amplitudes_train[k + self.D, :self.Nt], color="blue", marker=".", label='frame 2')
            ax.set_xticks([0, self.t[-1] / 2, self.t[-1]])
            ax.set_title(r'${mode}^{(' + str(k) + ')}$')
            ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
            ax.grid()
            ax.legend(loc='upper right')
        fig.supxlabel(r"time $t$")
        fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")
        fig.tight_layout()
        save_fig(filepath=impath + "time_amplitudes_" + "training", figure=fig)

    def plot_trainingshift(self, shifts_train, mu_vecs_train):
        fig, axs = plt.subplots(1, len(mu_vecs_train), figsize=(15, 6), num=2)
        plt.subplots_adjust(wspace=0)
        for k, ax in enumerate(axs):
            ax.plot(self.t, shifts_train[0][k * self.Nt:(k + 1) * self.Nt], color="red", marker=".", label='frame 1')
            ax.plot(self.t, shifts_train[1][k * self.Nt:(k + 1) * self.Nt], color="blue", marker=".", label='frame 2')
            ax.set_title(r'${\mu}^{(' + str(k) + ')}$')
            ax.grid()
            ax.legend(loc='center right')
        fig.supxlabel(r"time $t$")
        fig.supylabel(r"space $x$")
        fig.tight_layout()
        save_fig(filepath=impath + "shifts_" + "training", figure=fig)

    def plot_timeamplitudesPred(self, time_amplitudes_1_pred, time_amplitudes_1_test, time_amplitudes_2_pred,
                                time_amplitudes_2_test):
        # Frame 1
        fig, axs = plt.subplots(1, self.D, sharey=True, figsize=(18, 5), num=3)
        plt.subplots_adjust(wspace=0)
        for k, ax in enumerate(axs):
            ax.plot(self.t, time_amplitudes_1_pred[k, :], color="red", marker=".", label='predicted')
            ax.plot(self.t, time_amplitudes_1_test[k, :], color="blue", marker=".", label='actual')
            ax.plot(self.t, self.ta_pred_frame_wise[0][k, :], color="yellow", marker=".", label='interpolated')
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
        fig, axs = plt.subplots(1, self.D, sharey=True, figsize=(18, 5), num=4)
        plt.subplots_adjust(wspace=0)
        for k, ax in enumerate(axs):
            ax.plot(self.t, time_amplitudes_2_pred[k, :], color="red", marker=".", label='predicted')
            ax.plot(self.t, time_amplitudes_2_test[k, :], color="blue", marker=".", label='actual')
            ax.plot(self.t, self.ta_pred_frame_wise[1][k, :], color="yellow", marker=".", label='interpolated')
            ax.set_xticks([0, self.t[-1] / 2, self.t[-1]])
            ax.set_title(r'${mode}^{(' + str(k) + ')}$')
            ax.set_xticklabels(["0", r"$T/2$", r"$T$"])
            ax.legend(loc='upper right')
            ax.grid()
        fig.supxlabel(r"time $t$")
        fig.supylabel(r"coefficient $a_i^{f}(t,\mu)$")
        fig.tight_layout()
        save_fig(filepath=impath + "time_amplitudes_frame_2_" + "predicted", figure=fig)

    def plot_shiftsPred(self, shifts_1_pred, shifts_2_pred):
        # Frame 1
        fig, ax = plt.subplots(1, 1, figsize=(15, 6), num=5)
        plt.subplots_adjust(wspace=0)
        ax.plot(self.t, self.SHIFTS_TEST[0], color="red", marker=".", label='actual')
        ax.plot(self.t, shifts_1_pred.flatten(), color="blue", marker=".", label='predicted')
        ax.plot(self.t, self.delta_pred_frame_wise[0], color="yellow", marker=".", label='interpolated')
        ax.set_xlabel('t', fontsize=14)
        ax.set_ylabel('x', fontsize=14)
        ax.set_title(r'${frame}^{(' + str(1) + ')}$')
        ax.grid()
        ax.legend(loc='center right')
        fig.tight_layout()
        save_fig(filepath=impath + "shifts_frame_1" + "predicted", figure=fig)

        # Frame 2
        fig, ax = plt.subplots(1, 1, figsize=(15, 6), num=6)
        plt.subplots_adjust(wspace=0)
        ax.plot(self.t, self.SHIFTS_TEST[1], color="red", marker=".", label='actual')
        ax.plot(self.t, shifts_2_pred.flatten(), color="blue", marker=".", label='predicted')
        ax.plot(self.t, self.delta_pred_frame_wise[1], color="yellow", marker=".", label='interpolated')
        ax.set_xlabel('t', fontsize=14)
        ax.set_ylabel('x', fontsize=14)
        ax.set_title(r'${frame}^{(' + str(2) + ')}$')
        ax.grid()
        ax.legend(loc='center right')
        fig.tight_layout()
        save_fig(filepath=impath + "shifts_frame_2" + "predicted", figure=fig)
