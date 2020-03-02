from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier

class TableDataset(Dataset):
    def __init__(self, data, output_col=None):
        # Characterizes a Dataset for PyTorch
        self.n = data.shape[0]
        self.X = data.astype(np.float64).values
        if output_col:
            self.y = data[output_col].astype(np.float64).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

    def __len__(self):
        # Denotes the total number of samples.
        return self.n

    def __getitem__(self, idx):
        # Generates one sample of data.
        return [self.X[idx], self.y[idx]]

class Net(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, layer_dropouts):
        super().__init__()

        first_layer = nn.Linear(input_size, layer_sizes[0]) #59, 40
#         l1=nn.Linear(input_size, layer_sizes[0])
#         a1=nn.ReLU()
#         l2=nn.Linear(layer_sizes[0], layer_sizes[1])
        self.layers = nn.ModuleList([first_layer] +
                [nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]) #40, 65
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight.data)
        # output layers
        self.output_layer = nn.Linear(layer_sizes[-1], output_size)#65, 1
        nn.init.kaiming_normal_(self.output_layer.weight.data)
        # batch norm layers
        self.first_bn_layer = nn.BatchNorm1d(input_size) #32, 59
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in layer_sizes]) #32, 40 and 32, 65
        # dropout layers
        self.dropout_layers = nn.ModuleList([nn.Dropout(size) for size in layer_dropouts])

    def forward(self, data):
        x=self.first_bn_layer(data)
        for layer, dropout_layer, bn_layer in zip(self.layers, self.dropout_layers, self.bn_layers):
            x = F.relu(layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)
        x = F.sigmoid(self.output_layer(x))
        return x


# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, batchsize=640, no_of_epochs=10, lr=0.0005, input_size=59, layer_sizes=[30, 16], output_size=1, layer_dropouts=[0.01, 0.01], calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_median = None
        self.calibrate = calibrate_probabilities
        self.batchsize = batchsize
        self.epoch = no_of_epochs
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        self.net = Net(input_size=input_size, layer_sizes=layer_sizes, output_size=output_size, layer_dropouts=layer_dropouts)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here

        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        # ADD YOUR BASE CLASSIFIER HERE


    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
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
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE

        objects = []
        for col in X_raw.columns:
            if X_raw[col].dtype != np.float64 and X_raw[col].dtype != np.int64:
                objects.append(col)
        cat_features = []
        for obj in objects:
            if X_raw[obj].nunique() < 20:
                cat_features.append(obj)
        data_new = pd.concat([X_raw, pd.get_dummies(X_raw[cat_features], prefix=cat_features, dummy_na=True)],
                             axis=1).drop(cat_features, axis=1)
        other_objects = []
        for obj in objects[1:]:
            if obj not in cat_features:
                other_objects.append(obj)
        data_new[other_objects] = data_new[other_objects].astype('|S')
        label_encoders = {}
        for o in other_objects:
            label_encoders[o] = LabelEncoder()
            data_new[o] = label_encoders[o].fit_transform(data_new[o])
        X_clean = data_new.drop(['id_policy'], axis=1)
        X_clean.reset_index()
        X_clean = X_clean.fillna(0)
        data = pd.read_csv('part3_training_data.csv')
        X_clean = X_clean.join(data["made_claim"])
        return X_clean

    def fit(self, X_raw, y_raw, claims_raw):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model

        """
        nnz = np.where(claims_raw != 0)[0]
        self.y_median = np.median(claims_raw[nnz])
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        # if self.calibrate:
        #     self.base_classifier = fit_and_calibrate_classifier(
        #         self.base_classifier, X_raw, y_raw)
        # else:
        #     self.base_classifier = self.base_classifier.fit(X_raw, y_raw)

        X_clean = self._preprocessor(X_raw)
        y_clean = y_raw.values.reshape((-1, 1))
        X_train, X_val, y_train, y_val = train_test_split(X_clean, y_clean, test_size=0.1, random_state=42)
        output_feature = "made_claim"
        traindataset = TableDataset(X_train, output_feature)
        batchsize = 640
        traindataloader = DataLoader(traindataset, batchsize, shuffle=True, num_workers=1)
        self.losses = []
        for epoch in range(self.epoch):
            batch_losses = []
            self.net.train()
            for x, y in traindataloader:
                preds = self.net((x[:, :-1]).float())
                loss = self.criterion(preds, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                batch_losses.append(loss)
            average_batch_loss = sum(batch_losses) / len(batch_losses)
            self.losses.append(average_batch_loss)
        return self.net


    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        # X_clean = self._preprocessor(X_raw)
        X_clean = self._preprocessor(X_raw)
        output_feature = "made_claim"
        dataset = TableDataset(X_clean, output_feature)
        dataloader = DataLoader(dataset, self.batchsize, shuffle=True, num_workers=1)
        y_preds = []
        y_true = []
        self.net.eval()
        with torch.set_grad_enabled(False):
            acc_val = []
            for x_t, y_t in dataloader:
                val_preds = self.net((x_t[:, :-1]).float())
                y_true.append(y_t.numpy())
                y_preds.append(val_preds.numpy())
                try:
                    roc_auc = roc_auc_score(y_t, val_preds)
                    acc_val.append(roc_auc)
                except ValueError:
                    pass
        preds=[]
        y_preds=np.array(y_preds)
        for batch in y_preds:
            for x in batch.flatten():
                preds.append(float(x))
        preds = np.array(preds)
        return preds
        # return model# return probabilities for the positive class (label 1)

    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor
        self.y_median = self.y_median*0.97 # 0.97 has taken account of both the inflation and investment returns expected
        return self.predict_claim_probability(X_raw) * self.y_median

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part3_pricing_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model