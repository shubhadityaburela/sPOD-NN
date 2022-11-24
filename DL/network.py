# pytorch mlp for regression
import numpy as np
import torch.nn
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
import sys
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import ReLU, ELU, LeakyReLU
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn import L1Loss
from torch.nn.init import xavier_uniform_
import os
import time


sys.path.append('./DL_LIB/sPOD_DL_ROM/')

import Utilities

import matplotlib.pyplot as plt


def scale_params(PARAMS_TEST, params, scaling):

    if params['scaling']:
        # Reading the scaling factors for the testing data
        snapshot_max = scaling[0]
        snapshot_min = scaling[1]
        delta_max = scaling[2]
        delta_min = scaling[3]
        parameter_max = scaling[4]
        parameter_min = scaling[5]

        Utilities.scaling_componentwise_params(PARAMS_TEST, parameter_max, parameter_min,
                                               PARAMS_TEST.shape[0])

    return PARAMS_TEST


def scale_data(TA_TRAIN, params_train, params):

    num_samples = int(TA_TRAIN.shape[1])
    snapshot_max, snapshot_min = Utilities.max_min_componentwise(
        TA_TRAIN[:params['totalModes'], :],
        num_samples)
    modes_mat = TA_TRAIN[:params['totalModes'], :]
    Utilities.scaling_componentwise(modes_mat, snapshot_max, snapshot_min)
    TA_TRAIN[:params['totalModes'], :] = modes_mat

    delta_max = 0
    delta_min = 0
    if params['reduced_order_model_dimension'] != params['totalModes']:
        delta_max, delta_min = Utilities.max_min_componentwise(
            TA_TRAIN[params['totalModes']:, :],
            num_samples)
        delta_mat = TA_TRAIN[params['totalModes']:, :]
        Utilities.scaling_componentwise(delta_mat, delta_max, delta_min)
        TA_TRAIN[params['totalModes']:, :] = delta_mat

    parameter_max, parameter_min = Utilities.max_min_componentwise_params(params_train,
                                                                          num_samples,
                                                                          params_train.shape[0])
    Utilities.scaling_componentwise_params(params_train, parameter_max, parameter_min,
                                           params_train.shape[0])

    # Save the scaling factors for testing the network
    scaling = [snapshot_max, snapshot_min, delta_max, delta_min,
               parameter_max, parameter_min]

    return TA_TRAIN, params_train, scaling


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, ta_train, p_train, n_outputs):
        # store the inputs and outputs
        self.X = np.transpose(p_train.astype('float32'))
        self.y = np.transpose(ta_train.astype('float32'))
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), n_outputs))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.3):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs, n_outputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 25)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = ELU()
        # second hidden layer
        self.hidden2 = Linear(25, 50)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = ELU()
        # third hidden layer
        self.hidden3 = Linear(50, 75)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = ELU()
        # fourth hidden layer
        self.hidden4 = Linear(75, 50)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = LeakyReLU()
        # fourth layer and output
        self.hidden5 = Linear(50, n_outputs)
        xavier_uniform_(self.hidden5.weight)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # fourth hidden layer
        X = self.hidden4(X)
        X = self.act4(X)
        # fourth layer and output
        X = self.hidden5(X)
        return X


# prepare the dataset
def prepare_data(ta_train, p_train, n_outputs):
    # load the dataset
    dataset = CSVDataset(ta_train, p_train, n_outputs)
    # calculate split
    train, val = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=500, shuffle=True)
    val_dl = DataLoader(val, batch_size=500, shuffle=False)
    return train_dl, val_dl


# train the model
def train_model(train_dl, val_dl, n_outputs, model, epochs, lr=0.01, loss='L1'):
    # define the optimization
    if loss == 'L1':
        criterion = L1Loss()
    elif loss == 'MSE':
        criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    # enumerate epochs
    for epoch in range(epochs):
        # enumerate mini batches
        trainLoss = 0
        nBatches = 0
        model.train()
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

            trainLoss += loss.item()
            nBatches += 1

        model.eval()
        predictions, actuals = list(), list()
        for i, (inputs, targets) in enumerate(val_dl):
            # evaluate the model on the test set
            yhat = model(inputs)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            actual = actual.reshape((len(actual), n_outputs))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        if epoch % 500 == 0:
            num = np.sqrt(np.mean(np.linalg.norm(actuals - predictions, 2, axis=1) ** 2))
            den = np.sqrt(np.mean(np.linalg.norm(actuals, 2, axis=1) ** 2))
            rel_err = num / den
            print('Average loss at epoch {0} on training set: {1} and validation set: {2}'.
                  format(epoch, trainLoss / nBatches, rel_err))


# evaluate the model
def evaluate_model(val_dl, model, n_outputs):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(val_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), n_outputs))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)

    # calculate mse
    mse = mean_squared_error(actuals, predictions)

    num = np.sqrt(np.mean(np.linalg.norm(actuals - predictions, 2, axis=1) ** 2))
    den = np.sqrt(np.mean(np.linalg.norm(actuals, 2, axis=1) ** 2))
    rel_err = num / den

    return rel_err


def test_model(TA_TEST, params_test, trained_model=None, saved_model=True,
               PATH_TO_WEIGHTS='', params=None, scaling=None):

    if params['scaling']:
        # Reading the scaling factors for the testing data
        snapshot_max = scaling[0]
        snapshot_min = scaling[1]
        delta_max = scaling[2]
        delta_min = scaling[3]
        parameter_max = scaling[4]
        parameter_min = scaling[5]

    # test the model
    n_outputs = np.size(TA_TEST, 0)
    test_set = CSVDataset(TA_TEST, params_test, n_outputs)
    test_dl = DataLoader(test_set, batch_size=500, shuffle=False)

    numParams = np.size(params_test, 0)

    # define the network
    if saved_model:
        model = MLP(numParams, n_outputs)
        model.load_state_dict(torch.load(PATH_TO_WEIGHTS))
    else:
        model = trained_model

    model.eval()
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), n_outputs))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)

    if params['scaling']:
        modes_test_output = predictions[:, :params['totalModes']]
        Utilities.inverse_scaling_componentwise(modes_test_output,
                                                snapshot_max, snapshot_min)
        predictions[:, :params['totalModes']] = modes_test_output

        if params['reduced_order_model_dimension'] != params['totalModes']:
            delta_test_output = predictions[:, params['totalModes']:]
            Utilities.inverse_scaling_componentwise(delta_test_output,
                                                    delta_max, delta_min)
            predictions[:, params['totalModes']:] = delta_test_output

    # calculate mse
    mse = mean_squared_error(actuals, predictions)

    num = np.sqrt(np.mean(np.linalg.norm(actuals - predictions, 2, axis=1) ** 2))
    den = np.sqrt(np.mean(np.linalg.norm(actuals, 2, axis=1) ** 2))
    rel_err = num / den

    return rel_err, np.transpose(predictions)


def run_model(TA_TRAIN, params_train, epochs, lr, loss,
              logs_folder, pretrained_load=False, pretrained_weights='',
              params=None):

    log_folder = logs_folder + '/' + time.strftime("%Y_%m_%d__%H-%M-%S", time.localtime()) + '/'
    if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

    numParams = np.size(params_train, 0)
    numOutputs = np.size(TA_TRAIN, 0)

    scaling = 0
    if params['scaling']:
        TA_TRAIN, params_train, scaling = scale_data(TA_TRAIN, params_train, params)

    # prepare the data
    train_dl, val_dl = prepare_data(TA_TRAIN, params_train, numOutputs)

    # define the network
    model = MLP(numParams, numOutputs)

    # load pretrained weights
    if pretrained_load:
        model.load_state_dict(torch.load(pretrained_weights))

    # train the model
    train_model(train_dl, val_dl, numOutputs, model, epochs=epochs, lr=lr, loss=loss)

    # evaluate the model
    rel_err = evaluate_model(val_dl, model, numOutputs)
    print("Relative evaluation error :", rel_err)

    # Save the model
    log_folder_trained_model = log_folder + '/trained_weights/'
    if not os.path.isdir(log_folder_trained_model):
        os.makedirs(log_folder_trained_model)

    torch.save(model.state_dict(), log_folder_trained_model + 'weights.pt')

    log_folder_variables = log_folder + '/variables/'
    if not os.path.isdir(log_folder_variables):
        os.makedirs(log_folder_variables)
    np.save(log_folder_variables + 'scaling.npy', scaling, allow_pickle=True)

    return model, scaling
