from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

from part2_claim_classifier import ClaimClassifier

def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier

# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    # def __init__(self, batchsize=640, no_of_epochs=10, lr=0.0005, input_size=59, layer_sizes=[30, 16], output_size=1, layer_dropouts=[0.01, 0.01], calibrate_probabilities=False):
    def __init__(self, epoch=100, batchsize=32, learnrate=0.01, neurons=9, num_features=13, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities
        self.trained = False
        self.label_binarizer = {}

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
        self.base_classifier = ClaimClassifier(epoch, batchsize, learnrate, neurons, num_features)

    def _balance_dataset(self, X_y_raw):
        """Function to balance dataset used for training/validation/testing

        This function balances the dataset so it contains an equal number of
        Class 0 and Class 1 events

        Parameters
        ----------
        X_y_raw : ndarray
            An array, this is the raw data

        Returns
        -------
        X_y_balanced: ndarray
            An array, but balanced for each Class
        """
        # Seperate dataset into Class 0 and Class 1 events
        class_0 = X_y_raw[X_y_raw[:,-1] == 0]
        class_1 = X_y_raw[X_y_raw[:,-1] == 1]

        # Shuffle Class_0 events
        np.random.shuffle(class_0)

        # Take Subset of Class_0 events of equal size to Class 1 events
        class_1_size = class_1.shape[0]
        class_0_subset = class_0[:class_1_size,]
        X_y_balanced = np.vstack((class_0_subset,class_1))

        # Shuffle combined balanced dataset before returning
        np.random.shuffle(X_y_balanced)

        return X_y_balanced


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
        features_to_keep = ['pol_coverage', 'vh_age', 'vh_din', 'vh_fuel', 'vh_sale_begin', 'vh_sale_end', 'vh_speed', 'vh_weight']
        X_pre = X_raw[features_to_keep]

        for col in features_to_keep:

            if X_pre.dtypes[col] != 'float64' and X_pre.dtypes[col] != 'int64':

                X_pre[col].fillna("empty")

                if col not in self.label_binarizer.keys():
                    self.label_binarizer[col] = LabelBinarizer()

                if self.trained == False:
                    X_pre = X_pre.join(pd.DataFrame(self.label_binarizer[col].fit_transform(X_pre[col]),
                                                    columns=self.label_binarizer[col].classes_,
                                                    index=X_pre.index))
                else:
                    X_pre = X_pre.join(pd.DataFrame(self.label_binarizer[col].transform(X_pre[col]),
                                                    columns=self.label_binarizer[col].classes_,
                                                    index=X_pre.index))

                X_pre = X_pre.drop(columns=col)
            else:
                mean = np.nanmean(X_pre[col].values)
                X_pre[col].fillna(mean)

        return X_pre

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
        self.y_mean = np.mean(claims_raw[nnz])
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)
        X_Y_pandas = pd.concat([X_clean, y_raw], axis=1).reindex(X_clean.index)
        X_Y_clean = X_Y_pandas.to_numpy()

        X_Y_clean_balanced = self._balance_dataset(X_Y_clean)

        X_clean_balanced = pd.DataFrame(X_Y_clean_balanced[:,:-1])
        y_clean_balanced = pd.DataFrame(X_Y_clean_balanced[:,-1:])

        X_clean = X_clean_balanced
        y_raw = y_clean_balanced

        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        if self.calibrate:
            self.base_classifier = fit_and_calibrate_classifier(
                self.base_classifier, X_clean, y_raw)
        else:
            self.base_classifier = self.base_classifier.fit(X_clean, y_raw)

        self.trained = True
        return self.base_classifier

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

        X_clean = self._preprocessor(X_raw)
        return self.base_classifier.predict(X_clean)


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
        factor = 0.8 # 0.97 has taken account of both the inflation and investment returns expected
        return self.predict_claim_probability(X_raw) * self.y_median * factor

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        with open('part3_pricing_model.pickle', 'wb') as target:
            pickle.dump(self, target)


    def evaluate_architecture(self, X_test, Y_test):
        X = self._preprocessor(X_test)
        return self.base_classifier.evaluate_architecture(X, Y_test)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part3_pricing_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model
