# IMPORTS
# ______________________________________________________________________________________________________________________
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
import matplotlib
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
matplotlib.use('Agg')


# LIBRARY
# ______________________________________________________________________________________________________________________

def balance_train(train, balance):
    """
    :param train: the training dataset to resample
    :param balance: proportion of failed firm-year samples (e.g. 5%, 10% or 50%) in the resulting dataset
    :return: resampled train set by under-sampling the majority class
    """

    # shuffle rows
    train = train.sample(frac=1)
    # retain all failed firm-year samples, remove portion of healthy samples
    number_ones = len(train[train['label'] == 1])
    train = pd.concat([train[train['label'] == 1], train[train['label'] == 0][:(number_ones * balance)]])

    return train


def build_model(X_train, y_train, feature_constructor, k, C):
    """
    :param X_train: independent variables for training
    :param y_train: target for training
    :param feature_constructor: feature construction method
    :param k: number of features to retain after uni-variate feature selection
    :param C: inverse L2-regularisation strength
    :return: fitted model
    """

    # build sklearn pipelines for different modelling steps
    preprocessing = Pipeline([('feature_constructor', feature_constructor),
                              ('scaler', StandardScaler(with_mean=False))])
    feature_selection = Pipeline([('selector', SelectKBest(k=k))])
    classifier = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=1000)

    # combine the different steps in a single pipeline object
    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('feature_selection', feature_selection),
        ('classifier', classifier)])

    # fit the model and return the result
    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate(X_holdout, classifier):
    """
    :param X_holdout: holdout set for which we want to make predicions
    :param classifier: the trained model
    :return: returns predictions for both classes
    """

    # make and store the predictions
    predictions = classifier.predict_proba(X_holdout)
    predictions_0 = [row[0] for row in predictions]
    predictions_1 = [row[1] for row in predictions]

    return predictions_0, predictions_1


def get_feature_weights(classifier):
    """
    :param classifier: the trained model
    :return: the selected features and their relative importances
    """
    # retain the selected features
    features = classifier['preprocessing']['feature_constructor'].get_feature_names()
    selection_mask = classifier['feature_selection']['selector'].get_support()
    selected_features = [i for (i, v) in zip(features, selection_mask) if v]

    # retain the matching beta coefficients
    betas = classifier['classifier'].coef_[0]

    # store in dataframe and format results
    df = pd.DataFrame({'features': selected_features, 'betas': betas})
    df = df.drop(df[df['betas'] < 0.000001].index)
    df = df.sort_values('betas', ascending=False).reset_index(drop=True)

    return df
