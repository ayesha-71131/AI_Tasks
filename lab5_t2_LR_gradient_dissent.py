import pandas as pd
import numpy as np

# STEP 1: Gradient Descent Function
def gradient_descent(X1, X2, Y, learning_rate=0.01, epochs=1000):
    """
    Performs gradient descent to learn theta_0, theta_1, theta_2
    for: Y = θ0 + θ1*X1 + θ2*X2
    """
    n = len(Y)

    # Initialize thetas (weights)
    theta_0 = 0
    theta_1 = 0
    theta_2 = 0

    for epoch in range(epochs):
        # Predicted Y based on current thetas
        Y_pred = theta_0 + theta_1 * X1 + theta_2 * X2

        # Compute gradients (partial derivatives)
        d_theta_0 = (-2 / n) * np.sum(Y - Y_pred)
        d_theta_1 = (-2 / n) * np.sum((Y - Y_pred) * X1)
        d_theta_2 = (-2 / n) * np.sum((Y - Y_pred) * X2)

        # Update thetas
        theta_0 -= learning_rate * d_theta_0
        theta_1 -= learning_rate * d_theta_1
        theta_2 -= learning_rate * d_theta_2

    return theta_0, theta_1, theta_2

# STEP 2: Predict using learned parameters
def predict(X1, X2, theta_0, theta_1, theta_2):
    return theta_0 + theta_1 * X1 + theta_2 * X2

# STEP 3: Mean Squared Error
def calculate_mse(Y, Y_pred):
    return np.mean((Y - Y_pred) ** 2)

# STEP 4: Main Execution
def main():
    # Load dataset from CSV
    df = pd.read_csv("C:/Users/HP/OneDrive/Desktop/AI/AI post  mid/lab 5/housing.csv") 

    # Extract 2 numerical features and 1 target column
    X1 = df['area']
    X2 = df['bedrooms']
    Y = df['price']

    # Manually apply Min-Max scaling
    X1 = (X1 - X1.min()) / (X1.max() - X1.min())
    X2 = (X2 - X2.min()) / (X2.max() - X2.min())

    # Train model using gradient descent
    theta_0, theta_1, theta_2 = gradient_descent(X1, X2, Y)

    # Predict prices
    Y_pred = predict(X1, X2, theta_0, theta_1, theta_2)

    # Evaluate model
    mse = calculate_mse(Y, Y_pred)

    # Print results
    print("Learned Parameters:")
    print(f"theta_0 (intercept): {theta_0}")
    print(f"theta_1 (slope for area): {theta_1}")
    print(f"theta_2 (slope for bedrooms): {theta_2}")

    print("\nMean Squared Error (MSE):")
    print(mse)

main()
