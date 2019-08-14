import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def compute_cost(theta):
    predictions = np.dot(X, theta)
    squareError = np.square(predictions - y)
    #print(squareError)
    J = np.sum(squareError) / (2 * m)
    return J

	
def gradient_descent(w, alpha, iters):
    J_history = np.zeros([iters, 1])
    
    for iter in range(iters):
        gradient = np.dot(X, w) - y
        temp0 = w[0] - (alpha / m) * np.sum(gradient)
        temp1 = w[1] - (alpha / m) * np.sum(gradient * X[:, 1])
        w[0] = temp0
        w[1] = temp1
        
        #print(gradient)
        J_history[iter] = compute_cost(w)
    
    return w, J_history

	
df = pd.read_csv('data.txt', sep=',', header=None).values

#print(df.values)

x = np.array(df[:, 0])
y = np.array(df[:, 1])

m = len(y)

x.shape = (m, 1)
y.shape = (m, 1)

#print(m)

plt.plot(x, y, 'ro')
plt.xlabel('Profit')
plt.ylabel('Population of City')
plt.title('Linear Regression')
plt.show()

X = np.concatenate((np.ones([m, 1]), x), axis=1)
iterations = 1500
alpha = 0.01

theta = np.zeros([2, 1])
print(theta)
w, J_history = gradient_descent(theta, alpha, iterations)
print(w)










