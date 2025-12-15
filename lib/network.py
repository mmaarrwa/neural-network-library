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
        Main training loop with Mini-Batch Gradient Descent.
        """
        samples = len(x_train)
        loss_history = [] # 1. Initialize list to store loss values
        
        for i in range(epochs):
            # Optional: Shuffle data for better training
            indices = np.random.permutation(samples)
            x_train = x_train[indices]
            y_train = y_train[indices]
            
            epoch_loss = 0
            num_batches = 0
            
            # Mini-batch loop
            for j in range(0, samples, batch_size):
                # Create batch
                x_batch = x_train[j : j + batch_size]
                y_batch = y_train[j : j + batch_size]
                
                # Forward Pass
                output = self.predict(x_batch)
                
                # Calculate Loss
                loss = self.loss.forward(y_batch, output)
                epoch_loss += loss
                num_batches += 1
                
                # Backward Pass
                grad = self.loss.backward(y_batch, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
                    
                # Update Parameters
                for layer in self.layers:
                    if hasattr(layer, 'update'):
                        layer.update()
            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / num_batches
            
            # 2. Append to history list
            loss_history.append(avg_loss)
            
            print(f"Epoch {i+1}/{epochs} error={avg_loss:.6f}")
            
        # 3. RETURN the history list
        return loss_history