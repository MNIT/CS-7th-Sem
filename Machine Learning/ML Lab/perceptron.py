import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plotDecisionBoundary(param):
	plot_x = np.array([np.min(X[:,0]) - 2, np.max(X[:, 0]) + 2])
	
	plot_y = np.divide(np.add(param[0], (np.multiply(param[1], plot_x))), -param[2])
	
	plt.plot(plot_x, plot_y)

	plt.ylim(-1,10)
	plt.xlim(-1,10)

	
def plotGraph():
	pos = np.where(y == 1)[0]
	neg = np.where(y == 0)[0]
	
	plt.scatter(X[neg, 0], X[neg, 1], label= "0", color= "green", marker= "o")
	plt.scatter(X[pos, 0], X[pos, 1], label= "1", color= "blue", marker= "+")
	
	plt.xlabel('x - axis')
	plt.ylabel('y - axis')
	plt.title('Perceptron Algorithm')
	plt.legend()

	
def predict(x, params):
	activation = params[0]
	for i in range(len(x)):
		activation += params[i + 1] * x[i]
	return 1.0 if activation >= 0.0 else 0.0


def train_params(X, y, l_rate, iterations):
	params = np.zeros([1, len(X[0]) + 1], dtype = 'float')[0]
	

	for itr in range(iterations):
		sum_error = 0.0
		
		for i in range(len(X)):
			prediction = predict(X[i], params)
			error = y[i] - prediction
			sum_error += error**2
			params[0] = params[0] + l_rate * error
			
			for j in range(len(X[i])):
				params[j + 1] = params[j + 1] + l_rate * error * X[i, j]
		
		all_params.append(list(params))
		#plotDecisionBoundary(params)
		
		print('%d, error=%.2f' % (itr, sum_error))
	
	return params

	
def animate(i):
	print(i)
	plot_x = np.array([np.min(X[:,0]) - 2, np.max(X[:, 0]) + 2])
	plot_y = np.divide(np.add(all_params[i][0], (np.multiply(all_params[i][1], plot_x))), -all_params[i][2])
	
	#line.set_xdata(plot_x)
	line.set_ydata(plot_y)
	return line,


def init():
	plot_x = np.array([np.min(X[:,0]) - 2, np.max(X[:, 0]) + 2])
	plot_y = np.zeros([1, len(X[0])], dtype = 'float')[0]
	
	#line.set_xdata(plot_x)
	line.set_ydata(plot_y)
	return line,
	

X = np.array([[2.7, 2.5],
	[1.46, 2.36],
	[3.39, 4.40],
	[1.38, 1.85],
	[3.06, 3.00],
	[2.34, 3.56],
	[1.58, 2.94],
	[3.12, 1.23],
	[2.23, 0.45],
	[1.12, 0.12],
	[5.23, 2.42],
	[6.12, 0.23],
	[5.67, 0.56],
	[7.62, 2.75],
	[7.12, 1.12],
	[8.23, 2.34],
	[5.33, 2.08],
	[6.92, 1.77],
	[8.67, -0.24],
	[7.67, 3.50]])

y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

print('\nPlotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')

fig, ax = plt.subplots()
#line, = ax.plot([], [])
line, = ax.plot(np.array([np.min(X[:,0]) - 2, np.max(X[:, 0]) + 2]), np.zeros([1, len(X[0])], dtype = 'float')[0])
plotGraph()

l_rate = 0.03
iterations = 20
all_params = []
params = train_params(X, y, l_rate, iterations)

print()
print(params)

ani = animation.FuncAnimation(fig, animate, np.arange(0, 20), init_func=init, interval=200, blit=True)
plt.show()

print("\nExpected  Predicted")
for i in range(len(X)):
	prediction = predict(X[i], params)
	print("    %d         %d" % (y[i], prediction))


