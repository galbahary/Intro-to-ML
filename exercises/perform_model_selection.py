from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    x, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = x[:n_samples], y[:n_samples], x[n_samples:], y[n_samples:]

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_alphas = np.linspace(0.0000001, 0.01, num=n_evaluations)
    lasso_alphas = np.linspace(0.00001, 2.5, num=n_evaluations)
    ridge_scores = np.zeros((n_evaluations, 2))
    lasso_scores = np.zeros((n_evaluations, 2))
    for i in range(n_evaluations):
        ridge_model = RidgeRegression(ridge_alphas[i])
        lasso_model = Lasso(lasso_alphas[i], max_iter=5000)
        ridge_scores[i] = cross_validate(ridge_model, train_X, train_y, mean_square_error)
        lasso_scores[i] = cross_validate(lasso_model, train_X, train_y, mean_square_error)

    fig = make_subplots(1, 2, subplot_titles=[r"$\text{Ridge Regression}$", r"$\text{Lasso Regression}$"],
                        shared_xaxes=True) \
        .update_layout(title=r"$\text{Train and Validation Errors (averaged over the k-folds)}$", width=750, height=300) \
        .update_xaxes(title=r"$\lambda\text{ - Regularization parameter}$") \
        .add_traces([go.Scatter(x=ridge_alphas, y=ridge_scores[:, 0], name="Ridge Train Error"),
                     go.Scatter(x=ridge_alphas, y=ridge_scores[:, 1], name="Ridge Validation Error"),
                     go.Scatter(x=lasso_alphas, y=lasso_scores[:, 0], name="Lasso Train Error"),
                     go.Scatter(x=lasso_alphas, y=lasso_scores[:, 1], name="Lasso Validation Error")],
                    rows=[1, 1, 1, 1],
                    cols=[1, 1, 2, 2])
    fig.write_image("ridge_lasso_train_validation_errors.png")

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model

    reg_ridge = ridge_alphas[np.argmin(ridge_scores[:, 1])]
    reg_lasso = lasso_alphas[np.argmin(lasso_scores[:, 1])]
    print("CV-optimal regularization parameter -\n"
          "ridge: " + str(reg_ridge) + "\n"
                                       "lasso: " + str(reg_lasso) + "\n"
                                                                    "Test errors of models -\n"
                                                                    "Least-Squares: " + str(
        LinearRegression().fit(train_X, train_y).loss(test_X, test_y)) + "\n" +
          "Ridge: " + str(RidgeRegression(lam=reg_ridge).fit(train_X, train_y).loss(test_X, test_y)) + "\n"
                                                                                                       "Lasso: " + str(
        mean_square_error(test_y, Lasso(alpha=reg_lasso).fit(train_X, train_y).predict(test_X)))
          )


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
