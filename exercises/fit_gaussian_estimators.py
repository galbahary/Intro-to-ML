from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, 1000)
    ug = UnivariateGaussian().fit(X)
    print("(" + str(ug.mu_) + ", " + str(ug.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    diff_mu = np.zeros((100,))
    samples = np.zeros((100,))
    for i in range(101):
        if i == 0:
            continue
        samples[i - 1] = i * 10
        diff_mu[i - 1] = np.abs(10 - ug.fit(X[0:i * 10]).mu_)

    plot1 = go.Figure(data=[go.Scatter(
        x=samples,
        y=diff_mu,
        mode='markers', )
    ])
    plot1.update_layout(
        title="absolute distance between the estimated and true value of the expectation",
        xaxis_title="number of samples",
        yaxis_title="difference in expectation",
        legend_title="Legend Title",
    )
    plot1.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_res = ug.pdf(X)
    plot2 = go.Figure(data=[go.Scatter(
        x=X,
        y=pdf_res,
        mode='markers', )
    ])
    plot2.update_layout(
        title=" empirical PDF function under the fitted model",
        xaxis_title="X",
        yaxis_title="pdf value",
        legend_title="Legend Title",
    )
    plot2.show()
    # what are we expecting to see - the known "bell" shaped graph of pdf of normal distribution.


def test_multivariate_gaussian():

    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, sigma, 1000)
    ug = MultivariateGaussian().fit(X)
    print("(" + str(ug.mu_) + ", " + str(ug.cov_) + ")")

    # Question 5 - Likelihood evaluation
    log_likelihood_values = np.zeros((200, 200))
    indexes = np.linspace(-10, 10, 200)
    for i in range(len(indexes)):
        for j in range(len(indexes)):
            mu = np.array([indexes[i], 0, indexes[j], 0])
            log_likelihood_values[i][j] = MultivariateGaussian.log_likelihood(mu, sigma, X)
    plot1 = go.Figure(data=[go.Heatmap(
        x=indexes,
        y=indexes,
        z=log_likelihood_values)
    ])
    plot1.update_layout(
        title="Heatmap of log-likelihood as function of f1 and f3",
        xaxis_title="f1",
        yaxis_title="f3",
        legend_title="Legend Title",
    )
    plot1.show()
    # what can we learn from this graph - as closer we get to the point (0,4) of f1,f3 values, the value of the
    # log-likelihood is the best (highest).

    # Question 6 - Maximum likelihood
    f1_max, f3_max = np.unravel_index(np.argmax(log_likelihood_values, axis=None), log_likelihood_values.shape)
    print("the f1 value of max is: " + str(round(indexes[f1_max], 3)))
    print("the f3 value of max is: " + str(round(indexes[f3_max], 3)))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
