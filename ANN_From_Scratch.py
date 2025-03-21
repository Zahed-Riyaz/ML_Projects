import numpy as np
from layer import Layer


class ANN:
    def __init__(self):
        self.layers=[]

    def add_layer(self,input_size,num_neurons):
        self.layers.append(Layer(input_size,num_neurons))

    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def compute_loss_gradixent(self, y_pred, y_true):
        return 2 * (y_pred - y_true)

    def forward_prop(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward_prop(self, loss_gradient):
        for layer in reversed(self.layers):  # Go through layers in reverse order
            loss_gradient, dJ_dW, dJ_db = layer.backward(loss_gradient)
            learning_rate = 0.001
            # Update weights and biases
            layer.weights -= learning_rate * dJ_dW
            layer.bias -= learning_rate * dJ_db

    def train(self, X_train, y_train, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(X_train, y_train):
                y_pred = self.forward_prop(x)

                total_loss += self.compute_loss(y_pred, y)

                loss_gradient = self.compute_loss_gradient(y_pred, y)

                self.backward_prop(loss_gradient)

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(X_train)}")
