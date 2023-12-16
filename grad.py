# Gradient descent for Linear Regression
# yhat = wx + b
# loss = (y-yhat)**2/N
import numpy as np
#initialise some parameters
x = np.random.randn(10,1)
y = 2*x+np.random.rand()
# Parameters
w=0.0
b=0.0
# Hyperparameters
learning_rate = 0.01

# Create gradient descent function
def descent(x,y,w,b,learning_rate):
    dldw = 0.0
    dldb = 0.0
    N=x.shape[0]
    # loss = (y-(wx-b))**2
    for xi,yi in zip(x,y):
        dldw += -2*x*(y-(w*x+b))
        dldb += -2*(y-(w*x+b))
    
    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*dldb

    return w,b

# Iteratively make updates
for epoch in range(400):
    w,b = descent(x,y,w,b,learning_rate)
    yhat = w*x+b
    loss = np.divide(np.sum((y-yhat)**2,axis=0),x.shape[0]) 
    print(f'{epoch} loss is {loss[0]}, parameters w:{w[0]}, b:{b[0]}')





