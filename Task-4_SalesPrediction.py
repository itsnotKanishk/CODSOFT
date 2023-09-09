import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Load the advertising data from a CSV file
advertising_data = pd.read_csv("advertising.csv")

# Check for missing values in the dataset
print(advertising_data.isnull().sum())

# Separate the features (X) and the target variable (y)
X_features = advertising_data.drop(columns=['Sales'])
y_target = advertising_data['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, train_size=0.8, random_state=42)

# Create a linear regression model
linear_model = LinearRegression()

# Fit the model to the training data
linear_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = linear_model.predict(X_test)

# Calculate the mean squared error and R-squared score
mean_squared_error_value = mean_squared_error(y_test, y_pred)
r_squared_score = r2_score(y_test, y_pred)

# Print the results
print("R-squared Score (R2):", r_squared_score)
print("Mean Squared Error (MSE):", mean_squared_error_value)

# Create a scatter plot to visualize the actual vs. predicted sales
plt.scatter(y_test, y_test, color='green', label='Actual Sales')
plt.scatter(y_test, y_pred, color='skyblue', label='Predicted Sales')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual Sales vs. Predicted Sales")
plt.legend()
plt.show()
