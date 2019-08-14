import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Ellipse

def fun_y_ellipse(x, w0, w1, w2):
	t = list((w0 - w1 * x**2) / w2)
	
	for i in range(len(t)):
		if(t[i] > -10**(-9) and t[i]<0):
			t.remove(t[i])
			t.insert(i, 0)
	
	t = np.array(t)
	#print(t)
	
	return np.sqrt(t)

def fun_y_hyperbola(x, w0, w1, w2):
	t = list((w0 + w1 * x**2) / w2)
	
	for i in range(len(t)):
		if(t[i] > -10**(-9) and t[i]<0):
			t.remove(t[i])
			t.insert(i, 0)
	
	t = np.array(t)
	
	return np.sqrt(t)
	
def draw_ellipse(w0, w1, w2):
	a = math.sqrt(w0 / w1)
	b = math.sqrt(w0 / w2)
	
	x = np.arange(-a, a+0.1, 0.1)
	y = fun_y_ellipse(x, w0, w1, w2)
	#x = np.append(x, a)
	#y = np.append(y, 0)
	#print(x)
	#print(y)
	plt.plot(x, y, c='b')
	plt.plot(x, -y, c='b')
	maxn = np.max([a,b])
	plt.xlim(-maxn-4, maxn+4)
	plt.ylim(-maxn-4, maxn+4)
	plt.title("Equation : -|w0| + |w1| * x1^2 + |w2| * x2^2 = 0\nw0={0},  w1={1},  w2={2}".format(w0, w1, w2))
	plt.grid()
	plt.show()

def draw_vertical_line(w0, w1):
	if(w1 < 0):
		print("Wrong value of w1")
		return
	
	if(w0 > 0):
		print("Wrong value of w0, should be negative")
		return
		
	c = math.sqrt(-w0/w1)
	plt.axvline(x = c)
	plt.axvline(x = -c)
	plt.xlim(-c-4, c+4)
	plt.ylim(-c-4, c+4)
	plt.title("Equation : w0 + |w1| * x1^2 = 0\nw0={0},  w1={1}".format(w0, w1))
	plt.grid()
	plt.show()

	
def draw_hyperbola(w0, w1, w2):
	a = math.sqrt(abs(w0 / w1))
	b = math.sqrt(abs(w0 / w2))
	
	x = np.arange(-a-4, a+4, 0.1)
	y = fun_y_hyperbola(x, w0, w1, w2)
	
	#print(x)
	#print(y)
	plt.plot(x, y, c='b')
	plt.plot(x, -y, c='b')
	maxn = np.max([a,b])
	plt.xlim(-maxn-8, maxn+8)
	plt.ylim(-maxn-8, maxn+8)
	plt.title("Equation : w0 + |w1| * x1^2 - |w2| * x2^2 = 0\nw0={0},  w1={1},  w2={2}".format(w0, w1, w2))
	plt.grid()
	plt.show()
	
	
draw_ellipse(abs(36), abs(4), abs(9))
draw_ellipse(abs(36), abs(9), abs(4))

draw_vertical_line(-4, abs(1))

draw_hyperbola(-1, abs(1), abs(1))
draw_hyperbola(1, abs(1), abs(1))
draw_hyperbola(0, abs(1), abs(1))












