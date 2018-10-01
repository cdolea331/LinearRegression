from numpy import *
import matplotlib.pyplot as plt
def compute_loss(X, Y, W):
	N = float(len(X))
	return (1/(2*N)) * transpose(Y - (X @ W)) @ (Y - (X @ W)) 

def step_gradient(X, Y, W, learning_rate):
	gradients = zeros(len(W))
	N = float(len(X))
	gradients = (1/N) * transpose(X) @ (Y - (X @ W))
	W += learning_rate * gradients
	return W


def gradient_descent_runner(X, Y, W, learning_rate, iteration_count):
	for i in range(iteration_count): #Adjust weights according to # of time-steps
		weights = step_gradient(X, Y, W, learning_rate) #train weights(one step)
	return weights

def predict_dataset(X, W):
	return X @ W #Return linear combination of W * X(weights by inputs, then summed)


def run():
	inputs = genfromtxt('data.csv', delimiter = ',') #Read in the data from data.csv
	X = inputs[:,:-1] #All entries save the last are features, store them in a matrix m x (n - 1) where inputs is an mxn matrix
	Y = zeros((len(X), 1))#Initialize Y as a m x 1(m = # of entries) column vector
	Y[:,0] = array(inputs[:,-1]) #All results lie in the last column of the inputs, save them in a m x 1 column vector 
	X = insert(X, 0, 1, axis=1)
	W = zeros((len(X[0]), 1)) #Initilialize weights equal to the number of features in a column vector
	Wi = W.copy() # Copy initialized weights for comparison later
	learning_rate = 0.0001 #epsilon
	iteration_count = 100000 #Number of time-steps
	W = gradient_descent_runner(X, Y, W, learning_rate, iteration_count) #train weights, W = weight vector
	print("inital error = {0} \nWeights:\n{1}".format(compute_loss(X, Y, Wi), Wi))

	print("After {0} iterations: error: {1}, \nWeights:\n{2}".format(iteration_count, compute_loss(X, Y, W), W))
	predictions = predict_dataset(X, W)#Make predictions
	error = abs(predictions - Y)
	plt.plot(Y, color = 'black', linewidth = '1', label = 'Actual values' )#Plot real values
	plt.plot(predictions, color = 'blue', linewidth='1', label = 'Predicted values' )#Plot prediction model
	plt.plot(error, color = 'red', linewidth='1', label = 'Error' )#Plot prediction model
	plt.legend()
	plt.show()

if __name__== '__main__':
	run()
