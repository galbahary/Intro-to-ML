import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    weights_arr = []
    values_arr = []

    def callback(solver, weights, val, grad, t, eta, delta) -> None:
        values_arr.append(val)
        weights_arr.append(weights)

    return callback, values_arr, weights_arr


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for eta in etas:
        for module, name in [(L1, "L1"), (L2, "L2")]:
            callback, vals, weights = get_gd_state_recorder_callback()
            learning_rate = FixedLR(eta)
            GradientDescent(learning_rate=learning_rate, callback=callback).fit(module(init), None, None)

            fig = go.Figure([go.Scatter(y=vals, mode='markers')])
            fig.update_layout(
                xaxis_title="iterations", yaxis_title="convergence",
                title=f"for module {name} with learning rate {eta}"
            )
            fig.show()
            print(f"loss for eta: {eta} on module {name}: {np.min(vals)}")

            if eta == .01:
                fig = plot_descent_path(module, np.asarray(weights), title=f"for module: {name}, eta: {eta}")
                fig.show()


# *** optional ***
def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():

    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    gd = GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4))
    module = LogisticRegression(include_intercept=True).fit(X_train, y_train)
    y_prob = module.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color="red", )],

        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.show()

    best_alpha = thresholds[np.argmax(tpr - fpr)]
    # test_err = misclassification_error(y_test, module.predict_proba(X_test) >= best_alpha)
    test_err = LogisticRegression(include_intercept=True, alpha=best_alpha)\
        .fit(X_train, y_train).loss(X_test, y_test)
    print(f"The best alpha for TP & FP rations is: {best_alpha}, using it we get test error: {test_err}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter

    possible_lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for name in ["l1", "l2"]:
        validation_errs = []
        for lam in possible_lambdas:
            log_model = LogisticRegression(solver=gd, penalty=name, lam=lam)

            train_err, val_err = cross_validate(log_model, X_train, y_train, misclassification_error)
            validation_errs.append(val_err)

        min_err = np.min(validation_errs)
        best_lam_index = validation_errs.index(min_err)
        best_lam = possible_lambdas[best_lam_index]

        best_model = LogisticRegression(solver=gd, penalty=name, lam=best_lam)
        best_model.fit(X_train, y_train)

        test_err = misclassification_error(y_test, best_model.predict(X_test))

        print(f"Best lambda for {name} model is {best_lam} and best test error is {np.round(test_err, 2)}")


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
