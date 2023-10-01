# Gradient descent for Linear Regression
# yhat = wx + b
# loss = (y-yhat)**2/N
import numpy as np

# Initialize some parameters
x = np.random.randn(10, 1)
y = 5 * x + np.random.rand()

# Parameters
w = 0.0
b = 0.0

# Hyperparameters
learning_rate = 0.01
convergence_threshold = 1e-6  # Set a small threshold for convergence

# Initialize variables for tracking convergence
prev_loss = float('inf')

# Create gradient descent function
def descent(x, y, w, b, learning_rate):
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]
    
    for xi, yi in zip(x, y):
        dldw += -2 * xi * (yi - (w * xi + b))
        dldb += -2 * (yi - (w * xi + b))
    
    w = w - learning_rate * (1 / N) * dldw
    b = b - learning_rate * (1 / N) * dldb

    return w, b

# Initialize a counter for iterations
iterations = 0

# Training loop
while True:
    w, b = descent(x, y, w, b, learning_rate)
    yhat = w * x + b
    loss = np.divide(np.sum((y - yhat) ** 2, axis=0), x.shape[0])
    print(f'{iterations} loss is {loss[0]}, parameters w:{w[0]}, b:{b[0]}')
    
    # Check for convergence
    loss_change = prev_loss - loss
    
    # If loss change is small, consider it converged
    if abs(loss_change) < convergence_threshold:
        print("Converged - Steady State Reached")
        break
    
    prev_loss = loss
    iterations += 1
