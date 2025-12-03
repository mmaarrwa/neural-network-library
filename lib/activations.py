import numpy as np
# Importing the Base Layer class from the layers module
from layers import Layer 

## 1. Sigmoid Function
class Sigmoid(Layer):
    def forward(self, input):
        # [cite_start]Mathematical operation: f(x) = 1 / (1 + e^(-x)) [cite: 25]
        # Compresses input to a range between 0 and 1.
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    # Removed 'learning_rate' as optimization is handled separately
    def backward(self, output_gradient): 
        # Derivative: f'(x) = f(x) * (1 - f(x))
        sigmoid_derivative = self.output * (1 - self.output)
        # Apply Chain Rule
        return output_gradient * sigmoid_derivative

## 2. Tanh Function
class Tanh(Layer):
    def forward(self, input):
        # [cite_start]Mathematical operation: f(x) = tanh(x) [cite: 25]
        # Compresses input to a range between -1 and 1.
        self.output = np.tanh(input)
        return self.output

    # Removed 'learning_rate'
    def backward(self, output_gradient): 
        # Derivative: f'(x) = 1 - (f(x))^2
        tanh_derivative = 1 - np.power(self.output, 2)
        # Apply Chain Rule
        return output_gradient * tanh_derivative

## 3. ReLU Function
class ReLU(Layer):
    def forward(self, input):
        # [cite_start]Mathematical operation: f(x) = max(0, x) [cite: 25]
        self.input = input # Crucial for the derivative calculation in backward
        self.output = np.maximum(0, input)
        return self.output

    # Removed 'learning_rate'
    def backward(self, output_gradient): 
        # Derivative: 1 if input > 0, 0 otherwise
        relu_derivative = (self.input > 0) * 1  
        # Apply Chain Rule
        return output_gradient * relu_derivative

## 4. Softmax Function
class Softmax(Layer):
    def forward(self, input):
        # For numerical stability: subtract max(input)
        e_x = np.exp(input - np.max(input, axis=1, keepdims=True))
        # Mathematical operation: e^x / sum(e^x) (Creates a probability distribution)
        self.output = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.output

    # Removed 'learning_rate'
    def backward(self, output_gradient):
        # Calculate the Jacobian product for the gradient (simplified)
        input_gradient = np.zeros_like(output_gradient)
        for i in range(len(self.output)):
            # J is the Jacobian matrix
            J = np.diagflat(self.output[i]) - np.dot(self.output[i].reshape(-1, 1), self.output[i].reshape(1, -1))
            # Apply Chain Rule: dL/dx = J * dL/dy
            input_gradient[i] = np.dot(J, output_gradient[i])
        return input_gradient