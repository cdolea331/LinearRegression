from numpy import *
import matplotlib.pyplot as plt
def compute_loss(weights, points):
	loss = 0
	for i in range(len(points)):
		x = zeros(len(weights - 1))
		for j in range( len(weights) - 1):
			x[j] = points[i, j + 1]
		y = points[i, -1]
		loss += (y - (weights[0] + sum(multiply(x, weights[1:])))) ** 2
	return loss/float(len(points))

def step_gradient(weights, points, learning_rate):
	gradients = zeros(len(weights))
	N = float(len(points))
	for i in range(len(points)):   #TODO: fix stepping(indexing issue)
		x = zeros(len(weights) - 1)
		for j in range(len(weights) - 1):
			x[j] = points[i, j + 1]
		y = points[i, -1]
		gradients[0] += (-2/N) * (y - (weights[0] + sum(multiply(x, weights[1:]))))
		for j in range(1, len(weights)):
			gradients[i] += (-2/N) * (x[i] * (y - (weights[0] + sum(multiply(x, weights[1:])))))
	new_weights = zeros(len(weights))
	for i in range(len(weights)):
		new_weights[i] = weights[i] - (gradients[i] * learning_rate)
	return new_weights


def gradient_descent_runner(points, initial_weights, learning_rate, iteration_count):
	weights = zeros(len(initial_weights))
	for i in range(iteration_count): #Adjust weights according ot # of time-steps
		weights = step_gradient(weights, array(points), learning_rate) #train weights(one step)
	return weights

def predict_dataset(weights, points):
	predict = zeros(len(points))#initialize prediction array
	x = zeros(1, len(weights))
	for j in range(1, len(weights)):
		x[j - 1] = points[i, j]
	predict[i] = weights[0] + (sum(multiply(x, weights[1:])))
	return predict


def run():
	points = genfromtxt('data.csv', delimiter = ',') #Read in the data from data.csv(scores vs hours studied)
	learning_rate = 0.0001 #epsilon
	initial_weights = zeros(len(points[0]))#Initialize all weights as 0
	iteration_count = 10000 #Number of time-steps
	weights = gradient_descent_runner(points, initial_weights, learning_rate, iteration_count) #train weights
	print("inital error = {0} \nWeights:\n".format(compute_loss(initial_weights, points)))
	for i in range(initial_weights):
		print(initial_weights[i])

	print("After {0} iterations: error: {1}, \nWeights:\n".format(iteration_count, compute_loss(w0, w1, points)))
	for i in range(weights):
		print(weights[i])
	predictions = predict_dataset(weights, points)#Make predictions
	plt.scatter(points[:,0], points[:,1], color = 'black' )#Plot real values
	plt.plot(points[:,0], predictions, color = 'blue', linewidth='3' )#Plot prediction model
	plt.xticks(())
	plt.yticks(())
	plt.show()

if __name__== '__main__':
	run()