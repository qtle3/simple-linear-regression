# Salary Prediction using Simple Linear Regression

This project implements a simple linear regression model to predict the salary of an employee based on their years of experience. The model is trained on a dataset containing years of experience and corresponding salaries, and is capable of predicting salaries for new data points.

## Detailed Summary

The script loads the salary dataset, splits it into training and test sets, and trains a linear regression model using `scikit-learn`. It visualizes the relationship between years of experience and salary by plotting both the training and test data against the predicted regression line. The model's performance is evaluated on the test set, and it is also used to predict the salary for an employee with 12 years of experience. Additionally, the script outputs the model's coefficients and intercept, which provide insight into the linear relationship between the features and the target variable.

## Key Features

- **Data Handling:** Utilizes `pandas` for loading and manipulating the dataset.
- **Model Training:** Uses `scikit-learn` to train a simple linear regression model on the data.
- **Prediction:** Predicts employee salaries based on years of experience.
- **Visualization:** Visualizes the regression line against the actual data for both training and test sets.
- **Coefficients & Intercept:** Outputs the model's coefficients and intercept to understand the linear relationship.
