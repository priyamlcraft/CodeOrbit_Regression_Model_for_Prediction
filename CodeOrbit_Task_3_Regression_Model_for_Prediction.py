"""
CodeOrbit Task 3: Regression Model for Prediction
Uses the Diabetes dataset bundled with scikit-learn.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    # Load a sample regression dataset
    dataset = load_diabetes(as_frame=True)
    X = dataset.data
    y = dataset.target

    print("CodeOrbit Task 3: Regression Model for Prediction")
    print("-" * 55)
    print("Dataset shape:", X.shape)
    print("\nFirst five rows:")
    print(X.head())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.3f}")

    results = pd.DataFrame({
        "Actual Value": y_test.to_numpy(),
        "Predicted Value": predictions
    })
    print("\nActual vs Predicted (first 10 rows):")
    print(results.head(10).round(2))

    # Save an Actual vs Predicted plot
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, predictions, alpha=0.7)
    low = min(y_test.min(), predictions.min())
    high = max(y_test.max(), predictions.max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("CodeOrbit Task 3: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    print("\nPlot saved as actual_vs_predicted.png")
    plt.show()


if __name__ == "__main__":
    main()
