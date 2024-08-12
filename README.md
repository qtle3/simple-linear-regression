# Salary Prediction using Simple Linear Regression

This project implements a simple linear regression model to predict the salary of an employee based on their years of experience. The model is trained on a dataset containing years of experience and corresponding salaries, and is capable of predicting salaries for new data points.

## Detailed Summary

The script loads the salary dataset, splits it into training and test sets, and trains a linear regression model using `scikit-learn`. It visualizes the regression line against both training and test data to illustrate how well the model fits. The model predicts salaries for employees with 5 and 12 years of experience, rounding the results to two decimal places for clarity. Additionally, the script outputs the model's coefficients and intercept, which provide insights into the linear relationship between years of experience and salary.

## Key Features

- **Data Handling:** Utilizes `pandas` for loading and manipulating the dataset.
- **Model Training:** Uses `scikit-learn` to train a simple linear regression model on the data.
- **Prediction:** Predicts employee salaries based on years of experience.
- **Visualization:** Visualizes the regression line against the actual data for both training and test sets.
- **Coefficients & Intercept:** Outputs the model's coefficients and intercept to understand the linear relationship.
