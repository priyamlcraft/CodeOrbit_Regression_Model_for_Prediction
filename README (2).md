# CodeOrbit Task 3: Regression Model for Prediction

## Project Overview
This project builds a simple **Linear Regression** model using Python and Scikit-learn. It uses the Diabetes dataset included with Scikit-learn to predict a continuous target value and evaluate the model's performance.

## Objectives
- Load and explore a sample regression dataset.
- Split the dataset into training and testing sets.
- Train a Linear Regression model.
- Predict target values for the test set.
- Evaluate predictions using MAE, MSE, RMSE, and R².
- Visualize actual values against predicted values.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn

## Files
- `CodeOrbit_Task_3_Regression_Model_for_Prediction.py` — project source code.
- `requirements.txt` — required Python packages.
- `README.md` — project documentation.
- `actual_vs_predicted.png` — plot generated when the script runs.

## Installation
Install Python 3.10 or newer, then open a terminal in this project folder and run:

```bash
python -m pip install -r requirements.txt
```

## How to Run
```bash
python CodeOrbit_Task_3_Regression_Model_for_Prediction.py
```

The program prints the dataset preview, evaluation metrics, and a table comparing actual and predicted values. It also saves `actual_vs_predicted.png` in the current working directory.

## Evaluation Metrics
- **MAE:** average absolute difference between actual and predicted values.
- **MSE:** average squared prediction error.
- **RMSE:** square root of MSE, expressed in the target's units.
- **R²:** indicates how much variation in the target is explained by the model; values closer to 1 generally indicate a better fit.

## Note
This is an educational demonstration using a built-in sample dataset. Model results are for learning purposes and should not be treated as medical advice or used for clinical decisions.

## Internship
Prepared as part of the CodeOrbit Tech Machine Learning Internship — Task 3.
