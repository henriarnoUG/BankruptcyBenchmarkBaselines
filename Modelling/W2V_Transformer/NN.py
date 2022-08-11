# IMPORTS
# ______________________________________________________________________________________________________________________
from torch import nn
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import torch


# LIBRARY
# ______________________________________________________________________________________________________________________

class NeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_1_size, hidden_2_size, output_size, n, dropout_p):
        """
        :param input_size: the representation size (default 100 for w2v model and 768 for transformers)
        :param hidden_1_size: number of neurons in the first hidden layer
        :param hidden_2_size: number of neurons in the second hidden layer
        :param output_size: number of output neurons (default is 1 for classification with BCELoss)
        :param n: fixed-length time window
        :param dropout_p: dropout probability
        """
        super(NeuralNetwork, self).__init__()

        # initialize variables
        self.n = n
        self.input_size = input_size

        # initialize the FFNN layers
        self.linear_layers = nn.Sequential(

            # first hidden layer
            nn.Linear(input_size * self.n, hidden_1_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # second hidden layer
            nn.Linear(hidden_1_size, hidden_2_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),

            # output layer
            nn.Linear(hidden_2_size, output_size))

        # transform logits to probs
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        :param x: input to the neural network
        :return: class probability
        """
        # the input x has shape (batch_size, input_size, n)
        # we simply concat the encodings of consecutive documents when dealing with n > 1
        batch_size = x.shape[0]
        x = x.reshape([batch_size, self.n * self.input_size])

        # pass the encodings to the FFNN
        out = self.linear_layers(x)

        # squeeze output for BCELoss
        out = out.squeeze(1)

        # compute the probs through a sigmoid layer
        prob = self.sig(out)

        return prob


def train_loop(dataloader, model, loss_fn, optimizer):
    """
    :param dataloader: dataloader that allows running over the samples in batches
    :param model: the neural network to train
    :param loss_fn: the loss function
    :param optimizer: the optimizer
    :return: trains the NN based on the samples in the dataloader, returns the training loss
    """

    # specify number of samples in the dataloader
    size = len(dataloader.dataset)
    # specify the number of batches in the dataloader
    num_batches = len(dataloader)

    # initialise the training loss
    train_loss = 0

    # loop over the batches in the dataloader
    for current_batch, (X, y, idx) in enumerate(dataloader):

        # set model in train mode
        model.train()
        # forward pass
        pred = model(X)
        # compute loss
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        # set gradients to zero
        optimizer.zero_grad()
        # backward pass
        loss.backward()
        # adjust weights
        optimizer.step()

        # track progress
        if current_batch % 10 == 0:
            loss, current = loss.item(), current_batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # compute final training loss per batch
    train_loss /= num_batches

    return train_loss


def test_loop(dataloader, model, loss_fn):
    """
    :param dataloader: dataloader that allows running over the samples in batches
    :param model: the neural network to use in the evaluation
    :param loss_fn: the loss function
    :return: evaluates the samples in the dataloader according to the passed NN, returns the loss, auc performance
    metric and the predictions on the samples (with their index - used for checking the samples during analysis)
    """

    # set model to eval mode
    model.eval()

    # store number of batches in dataloader
    num_batches = len(dataloader)

    # init loss and lists to store prediction results
    test_loss = 0
    full_pred = []
    full_y = []
    indices = []

    # loop over the batches in the dataloader
    with torch.no_grad():
        for batch, (X, y, idx) in enumerate(dataloader):

            # generate predictions
            pred = model(X)
            # compute loss
            test_loss += loss_fn(pred, y).item()

            # store predictions, labels and sample indices of the batch
            full_pred.append(pred.detach().numpy())
            full_y.append(y.detach().numpy())
            indices.append(idx.detach().numpy())

    # format the predictions, labels and sample indices
    # allows computing performance metrics with sklearn library
    full_pred = np.concatenate(full_pred)
    full_y = np.hstack(full_y)
    indices = np.concatenate(indices)

    # compute auc
    auc = roc_auc_score(y_true=full_y, y_score=full_pred)
    # compute final loss per batch
    test_loss /= num_batches

    # print result
    print(100 * '_')
    print('Test AUC ' + str(auc))
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    return test_loss, auc, (indices, full_y, full_pred)

# ______________________________________________________________________________________________________________________
