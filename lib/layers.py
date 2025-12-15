import numpy as np

# Note: The 'learning_rate' parameter is removed from 'backward' to support 
# the separate optimizer design.

## 1. Base Layer Class (The Template)
class Layer:
    def __init__(self):
        # Stores input data for gradient calculation in the backward pass
        self.input = None
        # Stores output data for use in the next layer or activation derivative
        self.output = None

    # Forward Propagation (Must be implemented by children)
    def forward(self, input):
        raise NotImplementedError
        
    # Backward Propagation (Must be implemented by children)
    # Takes only the gradient from the subsequent layer
    def backward(self, output_gradient): 
        raise NotImplementedError

## 2. Dense Layer (Fully Connected) - Handles Gradients, NO UPDATE
class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Weights (W): Parameters to be learned
        #to solve the problem of the initialization
        #where in the xor problem multiplying with 0.01 was to small that leads to 
        #the error is constant and the model is not learning
        #while i the autoencoder where the input was 784 if we did not multiply with 0.01 the
        #wights was too large
        #xavier initialization is the solution for that
        scale = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(input_size, output_size) * scale
        # Biases (b): Parameters to be learned
        self.bias = np.zeros((1, output_size))
        
        # Optimizer management attributes
        self.optimizer = None 
        self.weights_gradient = None
        self.bias_gradient = None

    def set_optimizer(self, optimizer):
        """Assigns the optimizer object (e.g., SGD instance) to the layer."""
        self.optimizer = optimizer

    def forward(self, input):
        # Store input (X) for gradient calculation in backward pass
        self.input = input
        # Linear operation: Z = XW + b
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

    # Removed 'learning_rate' parameter
    def backward(self, output_gradient):
        # output_gradient is dL/dZ from the activation layer

        # 1. Calculate Gradients for W and b (Stored in self attributes)
        # dL/dW = X^T * dL/dZ
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        # dL/db = sum(dL/dZ)
        self.bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        # 2. UPDATE STEP IS ABSENT (Moved to the separate optimizer/update method)

        # 3. Calculate Gradient for the previous layer (dL/dX)
        input_gradient = np.dot(output_gradient, self.weights.T)
        return input_gradient
    
    def update(self):
        """Called by the Network class to execute the update via the optimizer."""
        if self.optimizer:
            # The optimizer applies the W_new = W_old - η * (∂L/∂W) formula
            self.optimizer.step(self.weights, self.weights_gradient)
            self.optimizer.step(self.bias, self.bias_gradient)