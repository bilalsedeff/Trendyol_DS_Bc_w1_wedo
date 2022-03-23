import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

cal_housing = fetch_california_housing()
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
X = X[['MedInc']]
y = cal_housing.target
fig = px.scatter(X, x="MedInc", y=y)
st.title("Scatter Plot")
st.plotly_chart(fig)
df = pd.DataFrame(dict(MedInc=X['MedInc'], Price=cal_housing.target))
st.write("df:", df)

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 0], df.iloc[:, 1], test_size=0.3, random_state=42)


def theta_checker(theta: int, error: float):
    if np.abs(error) < theta:
        return True
    else:
        return False


def error_calc(X: np.ndarray, y: np.ndarray, b0, b1):
    error = 0
    error = (y - (b0 + b1 * X)) ** 2
    return error


def loss_function(X: np.ndarray, y: np.ndarray, b0, b1, lambda_):
    loss = 0
    loss = np.sum((y - (b0 + b1 * X)) ** 2) + lambda_ * (b0 ** 2 + b1 ** 2)
    return loss


def grads(X: np.ndarray, y: np.ndarray, b0, b1, lambda_, theta):
    b0_grad = 0
    b1_grad = 0
    X = X.values
    y = y.values
    for i in range(X.shape[0]):
        if theta_checker(theta, error_calc(X[i], y[i], b0, b1)):
            b0_grad += -2 * (y[i] - (b0 + b1 * X[i])) + 2 * lambda_ * b0
            b1_grad += -2 * (y[i] - (b0 + b1 * X[i])) * X[i] + 2 * lambda_ * b1
        else:
            b0_grad += theta
            b1_grad += theta

    # for i in range(X.shape[0]):
    #

    return b0_grad, b1_grad


def regression(X: np.ndarray, y: np.ndarray, alpha: float, lambda_: float, epochs: int, batch_size: int, theta: float):
    beta = np.random.rand(2)
    st.write("Initial beta: ", beta)
    st.write("Initial loss: ", loss_function(X, y, beta[0], beta[1], lambda_))
    st.write("Initial gradient: ", grads(X, y, beta[0], beta[1], lambda_, theta))
    st.write("Starting sgd")
    my_bar = st.progress(0.)
    for i in range(epochs):
        # y_pred = beta[0] + beta[1] * X
        loss = loss_function(X, y, beta[0], beta[1], lambda_)
        st.markdown("---")
        st.write("Epoch: ", i, " Loss: ", loss)
        st.write("Gradient: ", grads(X, y, beta[0], beta[1], lambda_, theta))
        beta_prev = np.copy(beta)

        for j in range(batch_size):
            print("Epoch: ", i, " Batch: ", j, " Loss: ", loss, " Beta: ", beta)
            b0_grad, b1_grad = grads(X, y, beta[0], beta[1], lambda_, theta)
            beta[0] = beta[0] - alpha * b0_grad
            beta[1] = beta[1] - alpha * b1_grad
        if np.linalg.norm(beta - beta_prev) < 1e-4:
            st.write("Converged at epoch: ", i)
            print("Converged at epoch: ", i)
            break
        my_bar.progress(i / epochs)
    y_pred = beta[0] + beta[1] * X
    st.write("Final beta: ", beta)
    st.write("Final error: ", loss_function(X, y, beta[0], beta[1], lambda_))
    st.write("Final gradient: ", grads(X, y, beta[0], beta[1], lambda_, theta))
    st.write("Final MSE: ", mean_squared_error(y, y_pred))
    return beta


def plot_regression(X: np.ndarray, y: np.ndarray, beta: np.ndarray):
    y_pred = beta[0] + beta[1] * X
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y, mode="markers"))
    fig.add_trace(go.Scatter(x=X, y=y_pred, mode="lines"))
    st.plotly_chart(fig)


def main(verbose: bool = False):
    st.title("Housing Price Prediction")
    st.sidebar.subheader("Model Parameters")
    st.sidebar.markdown("---")
    lr = st.sidebar.slider("Learning Rate", 0.000001, 0.01, 0.000001, 0.0001)
    epochs = st.sidebar.slider("Epochs", 1, 10, 3, 1)
    batch_size = st.sidebar.slider("Batch Size", 1, 1000, 100, 1)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Hyperparameters")
    st.sidebar.markdown("---")
    theta = st.sidebar.slider("Theta", 1000, 10000, 1000, 100)
    lambda_ = st.sidebar.slider("Lambda", 0., 100., 1.)
    st.sidebar.markdown("---")
    st.sidebar.button("Verbose")

    if verbose:
        st.subheader("Data")
        st.subheader("X")
        st.write(X_train.head())
        st.subheader("y")
        st.write(y_train.head())

    beta = regression(X_train, y_train, lr, lambda_, epochs, batch_size, theta)
    plot_regression(X_test, y_test, beta)


if __name__ == "__main__":
    main()
