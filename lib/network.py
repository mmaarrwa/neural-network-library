import numpy as np

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        """Adds a layer to the network."""
        self.layers.append(layer)

    def use(self, loss, optimizer):
        """
        Sets the loss function and optimizer for the network.
        Also assigns the optimizer to each layer that needs it (like Dense).
        """
        self.loss = loss
        self.optimizer = optimizer
        
        # Link the optimizer to all layers that have weights (e.g., Dense)
        for layer in self.layers:
            if hasattr(layer, 'set_optimizer'):
                layer.set_optimizer(optimizer)

    def predict(self, input_data):
        """
        Runs the forward pass through all layers.
        Used for inference/testing.
        """
        # Iterate through all layers in order
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, epochs, batch_size=32):
        """
        Main training loop:
        1. Forward pass
        2. Calculate Loss
        3. Backward pass
        4. Update weights
        """
        # Determine number of batches
        samples = len(x_train)
        
        for i in range(epochs):
            loss = 0
            
            # Simple Batch Gradient Descent (or Full Batch if batch_size >= samples)
            # For a foundational library, we can loop slightly simpler:
            
            # 1. Forward Pass
            output = self.predict(x_train)
            
            # 2. Calculate Loss (for display)
            loss = self.loss.forward(y_train, output)
            
            # 3. Backward Pass
            # First, calculate the initial gradient from the loss function
            grad = self.loss.backward(y_train, output)
            
            # Then propagate backward through all layers (in reverse order)
            for layer in reversed(self.layers):
                grad = layer.backward(grad)
                
            # 4. Update Parameters
            # Call the update method on all layers (Dense layers will use the optimizer)
            for layer in self.layers:
                if hasattr(layer, 'update'):
                    layer.update()
            
            # Optional: Print progress every 100 epochs
            if (i + 1) % 100 == 0:
                print(f"Epoch {i+1}/{epochs} error={loss}")