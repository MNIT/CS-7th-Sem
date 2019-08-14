import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random as rn
xcor = []
ycor = []
def gen_points():
	global xcor,ycor
	m = rn.uniform(-1,1)
	ite = 20
	dirr = [-1,1]
	intercept = 10
	for i in range(ite):
		x = rn.uniform(-800,800)
		y = m*x+intercept
		di =  rn.choice(dirr)
		noise = rn.uniform(0,400)
		xcor.append(x)
		ycor.append(y+di*noise)

gen_points()
xcor.sort()
degree = 6
lambd = 8

error_array = []
degree_array = []
error_reg = []

def plot_graph(X_train, Y_train, theta):
	plt.plot(X_train[:, 1:2], Y_train[:, 0:1], 'ro')
	plt.xlabel('Profit')
	plt.ylabel('Population of city')
	plt.title('Linear Regression')
	plt.plot(X_train[:, 1:2], np.dot(X_train, theta))
	plt.show()

def plot_error(degree, error):
	plt.xlabel('degree')
	plt.ylabel('error')
	plt.title('Errors')
	plt.plot(np.asmatrix(degree).T, np.asmatrix(error).T)
	plt.show()

for k in range (1, degree + 1):
	X_train = np.concatenate((np.ones([np.shape(np.asmatrix(xcor).T)[0], 1]), np.asmatrix(xcor).T), axis = 1)
	Y_train = np.asmatrix(ycor).T

	for i in range(2, k + 1):
		X_train = np.concatenate((X_train, np.power(X_train[:, 1:2], i)), axis = 1)

	m = len(Y_train)
	print(np.shape(X_train))
	print(np.shape(Y_train))
	plt.plot(X_train[:, 1:2], Y_train[:, 0:1], 'ro')
	plt.xlabel('Profit')
	plt.ylabel('Population of city')
	plt.title('Linear regression')
	plt.show()



	'''NORMAL EQUATION'''
	theta = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T), Y_train)
	print(theta)
	plot_graph(X_train, Y_train, theta)
	#plot_curve(theta, 1)

	theta_reg = np.dot(np.dot(np.linalg.inv( (np.dot(X_train.T, X_train)) + lambd * np.identity(k + 1)), X_train.T), Y_train)
	print(theta_reg)
	plot_graph(X_train, Y_train, theta_reg)

	predictions = np.dot(X_train, theta) - Y_train
	sqError = np.multiply(predictions, predictions)
	J = np.sum(sqError) / (2 * m)
	print(J)

	predictions_reg = np.dot(X_train, theta_reg) - Y_train
	sqError_reg = np.multiply(predictions_reg, predictions_reg)
	J_reg = np.sum(sqError_reg) / (2 * m)
	print(J_reg)

	degree_array.append(k)
	error_array.append(J)
	error_reg.append(J_reg)

plot_error(degree_array, error_array)
plot_error(degree_array, error_reg)