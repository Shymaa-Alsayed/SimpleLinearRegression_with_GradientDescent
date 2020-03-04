# Libraries and packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def hypothesis(x, theta):
    """ Calculates the hypothesis function using x and theta """
    x = np.append(arr=np.ones((len(x), 1)).astype(int), values=x, axis=1)
    return x.dot(theta)


def cost(x, theta, y, m):
    return (0.5 / m) * sum((hypothesis(x, theta) - y) ** 2)


def gradient_descent(x, theta, y):
    updated_theta = theta
    for i in range(EPOCHS):
        temp0 = updated_theta[0] - ALPHA * (1 / m) * sum(hypothesis(x, updated_theta) - y)
        temp1 = updated_theta[1] - ALPHA * (1 / m) * sum(np.multiply(hypothesis(x, updated_theta) - y, x))
        updated_theta[0] = float(temp0)
        updated_theta[1] = float(temp1)
        print(updated_theta)
    return updated_theta


def predict(instance, optimum_parameters):
    instance = np.array([1, instance]).astype(float)
    prediction = instance.dot(optimum_parameters)
    return prediction


# Define number of epochs and learning rate
ALPHA = 0.01
EPOCHS = 1500

# Loading data
data = pd.read_csv('implementatindata.txt', header=None)
X = data.iloc[:, 0].values.reshape((-1, 1))
y = data.iloc[:, 1].values.reshape((-1, 1))

# Length of training examples
m = len(y)

# Initialize weights or theta0 and theta1
theta = np.zeros((2, 1)).astype(float)

# Compute hypothesis
h = hypothesis(X, theta)

# Compute cost function
initial_cost = cost(X, theta, y, m)
print('Initial cost : ', initial_cost)

# Running gradient descent
min_theta = gradient_descent(X, theta, y)

# Minimum cost
minimum_cost = cost(X, min_theta, y, m)
print('Minimum cost: ', minimum_cost)

## VISUALS ##

# Visualizing our raw data
plt.scatter(X, y)
plt.show()

# Visualizing hypothesis on the training set
predictions_of_training_set = []
for i in range(m):
    predictions_of_training_set.append(float(predict(X[i], min_theta)))
plt.scatter(X, y)
plt.plot(X, predictions_of_training_set, color='green')
plt.show()

## Predict new instance ##
new_instance = 17
print('Predicted value: ', predict(new_instance, min_theta))

