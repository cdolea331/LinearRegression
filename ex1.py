from numpy import *
import matplotlib.pyplot as plt
def compute_loss(w0, w1, points):
	loss = 0
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		loss += (y - (w1 * x + w0 )) ** 2
	return loss/float(len(points))

def step_gradient(w0, w1, points, learning_rate):
	grad_w0 = 0
	grad_w1 = 0
	N = float(len(points))
	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1]
		grad_w0 -= (y - (w1 * x + w0 ))
		grad_w1 -= x * (y - (w1 * x + w0 ))
	new_w0 = w0 -((2/N) * grad_w0 * learning_rate)
	new_w1 = w1 -((2/N) * grad_w1 * learning_rate)
	return [new_w0, new_w1]


def gradient_descent_runner(points, initial_w0, initial_w1, learning_rate, iteration_count):
	w0 = initial_w0 #Initialize bias to given bias
	w1 = initial_w1 #Initialize w1 to given w1
	for i in range(iteration_count): #Adjust weights according ot # of time-steps
		w0, w1 = step_gradient(w0, w1, array(points), learning_rate) #train weights(one step)
	return [w0,w1]

def predict_dataset(w0, w1, points):
	predict = zeros(len(points))#initialize prediction array
	for i in range(len(points)):#make predictions
		x = points[i,0]
		predict[i] = w0 + (x * w1)
	return predict


def run():
	points = genfromtxt('data.csv', delimiter = ',') #Read in the data from data.csv(scores vs hours studied)
	learning_rate = 0.0001 #epsilon
	weights = zeros(len(points[0]))#Initialize all weights as 0
	print(len(points[0]))
	initial_w0  = 0 #Initialize bias
	initial_w1 = 0 #Initialize w1(hours studied)
	iteration_count = 10000 #Number of time-steps
	[w0,w1] = gradient_descent_runner(points, initial_w0, initial_w1, learning_rate, iteration_count) #train weights
	print("inital error = {0}, initial w0 = {1}, initial w1 = {2}".format(compute_loss(initial_w0, initial_w1, points), initial_w0, initial_w1))
	print("After {0} iterations: error: {1}, w0: {2}, w1: {3}".format(iteration_count, compute_loss(w0, w1, points), w0, w1))
	predictions = predict_dataset(w0, w1, points)#Make predictions
	plt.scatter(points[:,0], points[:,1], color = 'black' )#Plot real values
	plt.plot(points[:,0], predictions, color = 'blue', linewidth='3' )#Plot prediction model
	plt.xticks(())
	plt.yticks(())
	plt.show()

if __name__== '__main__':
	run()