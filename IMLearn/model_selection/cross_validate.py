from __future__ import annotations

import random
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score = 0.0
    validation_score = 0.0
    ids = np.arange(X.shape[0])
    random.shuffle(ids)
    splited = np.array_split(ids, cv)
    for fold in splited:
        test_X = np.array(X)[fold]
        test_y = np.array(y)[fold]
        train_y = deepcopy(y)
        train_y = np.delete(train_y, fold)
        train_X = deepcopy(X)
        train_X = np.delete(train_X, fold, axis=0)
        fitted_est = deepcopy(estimator).fit(train_X, train_y)
        train_score += scoring(train_y, fitted_est.predict(train_X))
        validation_score += scoring(test_y, fitted_est.predict(test_X))
    return train_score / cv, validation_score / cv
