import numpy as np

class Layer:
	#handles the big operations for singular layers. 
	#You'll probably find functions from this class inside for loops in ANN.
	def __init__(self,input_size,output_size):
		self.input_size=input_size
		self.output_size=output_size
		# Here, output size is simply the output of one layer.
		# Input will be the number of neurons of prev layer
		self.weights=np.random.random([output_size,input_size])
		self.bias=np.random.random([output_size,1])


	def sigmoid(self,x):
		return 1/(1+np.exp(-x))

	def derivative_sigmoid(self,x):
		return x/(1-x)

	def forward(self,input_data):
		self.input_data=input_data
		self.z= np.dot(self.weights,input_data,)+self.bias
		return self.sigmoid(self.z)

	def backward(self, dJ_da):
		dJ_dz = dJ_da * self.derivative_sigmoid(self.z)

		# Computing the derivative of the cost function w.r.t. weights, biases, and inputs
		dJ_dW = np.dot(dJ_dz, self.input_data.T)
		dJ_db = np.sum(dJ_dz, axis=1, keepdims=True)
		dJ_du = np.dot(self.weights.T, dJ_dz)
		        
		return dJ_du, dJ_dW, dJ_db


