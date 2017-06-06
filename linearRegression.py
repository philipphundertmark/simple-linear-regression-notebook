import matplotlib.pyplot as plt
from numpy import *

def compute_cost(b, m, points):
	total_cost = 0
	N = float(len(points))
	#Compute sum of squared errors
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		total_cost += (y - (m * x + b)) ** 2
	#Return average of squared error
	return total_cost/N

def step_gradient(b_current, m_current, points, learning_rate):
	m_gradient = 0
	b_gradient = 0
	N = float(len(points))

	#Calculate Gradient
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		m_gradient += - x * (y - (m_current * x + b_current))
		b_gradient += - (y - (m_current * x + b_current))

	#Update current m and b
	m_updated = m_current - learning_rate * ((2/N) * m_gradient)
	b_updated = b_current - learning_rate * ((2/N) * b_gradient)

	#Return updated parameters
	return b_updated, m_updated


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m
	cost_graph = []

	#In every iteration: optimize b, m and compute it's cost
	for i in range(num_iterations):
		cost_graph.append(compute_cost(b, m, points))
		b, m = step_gradient(b, m, array(points), learning_rate)

	return [b, m, cost_graph]

def run():
	points = genfromtxt('data.csv', delimiter=',')

	#hyperparamters
	learning_rate = 0.0001
	initial_b = 0
	initial_m = 0
	num_iterations = 10

	#Calculate b, m to best fit the given data
	[b, m, cost_graph] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

	#Print optimized parameters
	print 'Optimized b:', b
	print 'Optimized m:', m

	#Print error with optimized parameters
	print 'Minimized cost:', compute_cost(b, m, points)

	#Plot cost per iteration
	print 'Plotting cost per iteration...'
	plt.plot(cost_graph)
	plt.xlabel('No. of iterations')
	plt.ylabel('Cost')
	plt.title('Cost after each iterations')
	plt.show()

	plt.clf()

	print 'Plotting dataset...'

	x = array(points[:,0])
	y = array(points[:,1])
	#Plot dataset
	plt.scatter(x, y)
	#Predict y values
	pred = m * x + b
	#Plot predictions as line of best fit
	plt.plot(x, pred, c='r')
	plt.xlabel('Hours of study')
	plt.ylabel('Test scores')
	plt.title('Predicting test scores')
	plt.show()


if __name__ == '__main__':
	run()

