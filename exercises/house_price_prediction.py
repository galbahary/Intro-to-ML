from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

mean_per_feature = {}
positive_value_check = ["bedrooms", "bathrooms"]
non_negative_value_check = ["sqft_living", "sqft_lot", "sqft_above", "sqft_lot15", "sqft_living15"]


def extra_process_for_train(X: pd.DataFrame):

    price_filters = X.loc[(X.price <= 0) | (X.price.isnull())]
    X.drop(price_filters.index, inplace=True)

    for f in positive_value_check:
        mean_per_feature[f] = X.loc[X[f] > 0, f].mean()
        X.loc[(X[f] <= 0), f] = mean_per_feature[f]
    for f in non_negative_value_check:
        mean_per_feature[f] = X.loc[X[f] >= 0, f].mean()
        X.loc[(X[f] < 0), f] = mean_per_feature[f]

    return X


def extra_process_for_test(X: pd.DataFrame):
    for f in positive_value_check:  # replacement of illegal values with the mean to avoid noise
        X.loc[(X[f] <= 0), f] = mean_per_feature[f]

    for f in non_negative_value_check:  # replacement of illegal values with the mean to avoid noise
        X.loc[(X[f] < 0), f] = mean_per_feature[f]

    return X


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    X.fillna(0)
    if y is not None:
        X.insert(0, "price", y)
        X.drop_duplicates(inplace=True)
        X = extra_process_for_train(X)
    else:
        X.fillna(0)
        X = extra_process_for_test(X)

    X.drop(columns=['id', 'date', 'lat', 'long'], inplace=True)
    X = pd.get_dummies(X, columns=['zipcode'])

    if y is not None:
        return X.drop(columns='price'), X["price"]
    return X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X:
        plot1 = go.Figure(data=[go.Scatter(
            x=X[feature],
            y=y,
            mode='markers', )
        ])
        plot1.update_layout(
            title="Pearson Correlation : " + str((np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y)))),
            xaxis_title=feature,
            yaxis_title="price",
            legend_title="Legend Title",
        )
        plot1.write_image(file=output_path + str(feature) + ".png", format="png")


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    train_x, train_y, test_x, test_y = split_train_test(df.drop(["price"], axis=1), df["price"])

    # Question 2 - Preprocessing of housing prices dataset
    train_x, train_y = preprocess_data(train_x, train_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_x, train_y, "")

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    test_x.insert(0, "price", test_y)  # removing lines with illegal price value from test
    test_x = test_x.drop(test_x.loc[(test_x.price <= 0) | (test_x.price.isnull())].index)
    test_x, test_y = test_x.drop(columns='price'), test_x["price"]
    test_x = preprocess_data(test_x, None)
    test_x = test_x.reindex(columns=train_x.columns, fill_value=0)  # make sure cols are the same

    results = np.zeros((91, 10))
    for percent in range(10, 101):
        for it in range(10):
            sample_train_x = train_x.sample(frac=(percent / 100))
            sample_train_y = train_y.loc[sample_train_x.index]
            fitted = LinearRegression(include_intercept=True).fit(sample_train_x.to_numpy(),
                                                                  sample_train_y.to_numpy())
            loss_val = fitted.loss(test_x.to_numpy(), test_y.to_numpy())
            results[percent - 10, it] = loss_val
    mean, var = results.mean(axis=1), results.std(axis=1)
    ps = list(range(10, 101))
    plot2 = go.Figure([go.Scatter(x=ps, y=mean - 2 * var, fill=None, mode="lines", line=dict(color="green")),
                       go.Scatter(x=ps, y=mean + 2 * var, fill=None, mode="lines", line=dict(color="blue")),
                       go.Scatter(x=ps, y=mean, mode="markers+lines", marker=dict(color="black"))],
                      layout=go.Layout(title="MSE as Function Of Training sample percentage",
                                       xaxis=dict(title="Percentage of sample from Training Set"),
                                       yaxis=dict(title="MSE"),
                                       showlegend=False))
    plot2.show()
