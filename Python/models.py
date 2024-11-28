# Importing necessary libraries
import pystan
from skgarden import RandomForestQuantileRegressor
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. Bayesian Regression
def bayesian_regression():
    model_code = """
    data {
        int<lower=0> N;
        vector[N] x;
        vector[N] y;
    }
    parameters {
        real alpha;
        real beta;
        real<lower=0> sigma;
    }
    model {
        y ~ normal(alpha + beta * x, sigma);
    }
    """
    stan_model = pystan.StanModel(model_code=model_code)
    data = {
        'N': 100,
        'x': np.arange(100),
        'y': np.arange(100) * 2 + 3 + np.random.normal(0, 5, 100)
    }
    fit = stan_model.sampling(data=data, iter=1000, chains=4)
    print(fit)

# 2. Quantile Regression Forests
def quantile_regression_forests():
    X = np.random.rand(100, 1)
    y = X.ravel() + np.random.normal(0, 0.1, 100)
    qrf = RandomForestQuantileRegressor(n_estimators=100)
    qrf.fit(X, y)
    print("Quantile prediction (95%):", qrf.predict(X, quantile=95))

# 3. Neural Network Regression
def neural_network_regression():
    X = np.random.rand(100, 1)
    y = 2 * X + 1 + np.random.normal(0, 0.1, (100, 1))
    model = Sequential([
        Dense(64, activation='relu', input_shape=(1,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=10)
    print("Prediction:", model.predict([[0.5]]))

# Run all models
bayesian_regression()
quantile_regression_forests()
neural_network_regression()
