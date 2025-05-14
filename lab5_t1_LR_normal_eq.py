import pandas as pd
import numpy as np

def load_data(filename):
    df = pd.read_csv(filename)
    X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
    Y = df['price']
    return X, Y

def preprocess_features(X):
    X_scaled = X.copy()
    for col in X.columns:
        min_val = X[col].min()
        max_val = X[col].max()
        X_scaled[col] = (X[col] - min_val) / (max_val - min_val)
    return X_scaled


def add_bias_term(X):
    n = X.shape[0]  # Number of rows/samples
    ones = np.ones((n, 1))  # Column of 1s
    X_b = np.hstack((ones, X.values))  # Combine with original features
    return X_b


def compute_theta(X_b, Y):
    X_transpose = X_b.T
    theta = np.linalg.inv(X_transpose.dot(X_b)).dot(X_transpose).dot(Y)
    return theta


def predict(X_b, theta):
    return X_b.dot(theta)


def calculate_mse(Y, Y_pred):
    errors = Y - Y_pred
    mse = np.mean(errors ** 2)
    return mse


def main():
    X, Y = load_data("C:/Users/HP/OneDrive/Desktop/AI/AI post  mid/lab 5/housing.csv")
    X_scaled = preprocess_features(X)
    X_b = add_bias_term(X_scaled)
    theta = compute_theta(X_b, Y)
    Y_pred = predict(X_b, theta)
    mse = calculate_mse(Y, Y_pred)

    print("Learned Parameters (Theta):")
    print(theta)

    print("\nMean Squared Error (MSE):")
    print(mse)

main()
