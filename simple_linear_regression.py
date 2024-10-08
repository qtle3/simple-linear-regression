# Description: Simple linear regression model to predict the salary of an employee based on the years of experience.
# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(dataset)

# split the dataset into training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# train the simple linear regression model on the training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict the test set results
y_pred = regressor.predict(X_test)

# visualize the training set results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="green")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# visualize the test set results
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="green")
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# predict the salary of an employee with 5 years of experience
mid_level = round(regressor.predict([[5]])[0], 2)
print("Salary prediction for a mid level role:", mid_level)

# predict the salary of an employee with 12 years of experience
senior_level = round(regressor.predict([[12]])[0], 2)
print("Salary prediction for a senior level role:", senior_level)

# get the coefficients and intercept
print(round(regressor.coef_[0], 2))
print(round(regressor.intercept_, 2))
