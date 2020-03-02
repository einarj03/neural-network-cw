import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ClaimClassifier(nn.Module):

    def __init__(self, lin_layer_sizes, output_size, lin_layer_dropouts, emb_dims=None, no_of_cont=None,
                 emb_dropout=None, cat_cols=None):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        super().__init__()

        self.cat_cols = cat_cols if cat_cols else []
        self.con_cols = []
        self.y = []

        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self.no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_cont = no_of_cont

        first_lin_layer = nn.Linear(self.no_of_embs+self.no_of_con, lin_layer_sizes[0])

        self.lin_layers = \
            nn.ModuleList([first_lin_layer] +
                [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i+1])
                    for i in range(len(lin_layer_sizes)-1)])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        self.first_bn_layer = nn.BatchNorm1d(self.no_of_con)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        self.emb_dropout_layer = nn.Dropout(emb_dropout)

        self.dropout_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])


    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """
        # YOUR CODE HERE
        self.n = X_raw.shape[0]

        self.con_cols = [col for col in X_raw.columns if col not in self.cat_cols + [self.output_col]]

        if self.con_cols:
            self.con_X = X_raw[self.con_cols].astype(np.float32).values
        else:
            self.con_X = np.zeros((self.n, 1))

        if self.cat_cols:
            self.cat_X = X_raw[self.cat_cols].astype(np.int64).values
        else:
            self.cat_X = np.zeros((self.n, 1))

        return [self.cat_X, self.con_X]

    def fit(self, X_raw, y_raw=None):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)
        # YOUR CODE HERE

        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self.no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_cont = no_of_cont

        first_lin_layer = nn.Linear(self.no_of_embs+self.no_of_con, lin_layer_sizes[0])

        self.lin_layers = \
            nn.ModuleList([first_lin_layer] +
                [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i+1])
                    for i in range(len(lin_layer_sizes)-1)])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        self.first_bn_layer = nn.BatchNorm1d(self.no_of_con)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        self.emb_dropout_layer = nn.Dropout(emb_dropout)

        self.dropout_layers = nn.ModuleList([nn.Dropout(size) for size in lin_layer_dropouts])


    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)

        # YOUR CODE HERE
        cat_data = self._preprocessor(X_raw)[0]
        cont_data = self._preprocessor(X_raw)[1]

        if y_raw:
            self.y = X_raw[self.output_col].values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

        if self.no_of_embs != 0:
            x = [emb_layer(cat_data[:, i])
                 for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)
        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)
            if self.no_of_embs != 0:
                x = torch.cat([x, normalized_cont_data], 1)
            else:
                x = normalized_cont_data
        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.dropout_layers, self.bn_layers):
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)

        return x

      # YOUR PREDICTED CLASS LABELS

    def evaluate_architecture(self):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        pass

    def save_model(self):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model

# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch():
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """

    return  # Return the chosen hyper parameters
