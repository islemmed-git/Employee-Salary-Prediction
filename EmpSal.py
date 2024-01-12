import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Creating a random dataset
np.random.seed(42)
data = pd.DataFrame({
    'experience': np.random.randint(1, 20, 100),
    'education': np.random.randint(1, 5, 100),
    'position': np.random.choice(['junior', 'senior'], 100),
    'salary': 3000 + 100 * np.random.randn(100)
})

# Display the first few rows of the dataset
print(data.head())

# Features (X) and Target (y)
X = data[['experience', 'education', 'position']]
y = data['salary']

# Convert categorical variables into numerical representations (one-hot encoding)
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize actual vs. predicted salaries
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs. Predicted Employee Salaries')
plt.show()
