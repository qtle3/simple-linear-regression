# Description: Simple linear regression model to predict the salary of an employee based on the years of experience.
# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# split the dataset into training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
