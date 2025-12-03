import numpy as np

class Loss_MSE:
    
    # Note: This class does not inherit from Layer because it is not 
    # a sequential component of the network's layer chain.

    def _init_(self):
        # The loss function does not need any parameters to store 
        # (no W or b)
        pass

    # Forward method: Calculates the loss value (L)
    # The mathematical formula is: L = (1/N) * SUM((Y_true - Y_pred)^2)
    def forward(self, y_true, y_pred):
        """
        Calculates the Mean Squared Error (MSE) loss value.
        
        Args:
            y_true (np.array): The true labels/targets.
            y_pred (np.array): The network's predictions.
        
        Returns:
            float: The single scalar value of the calculated loss.
        """
        # np.power(a, 2) is used for squaring the error
        # np.mean divides by N (number of samples)
        return np.mean(np.power(y_true - y_pred, 2))

    # Backward method: Calculates the initial gradient (∂L/∂Y_pred)
    def backward(self, y_true, y_pred):
        """
        Calculates the initial gradient of the loss with respect to the output 
        (dL/dY_pred). This gradient starts the backpropagation chain.
        
        Args:
            y_true (np.array): The true labels.
            y_pred (np.array): The network's predictions.
            
        Returns:
            np.array: The initial gradient array.
        """
        # N is the batch size (number of samples)
        N = y_true.shape[0]
        
        # Derivative of MSE: dL/dY_pred = (-2/N) * (Y_true - Y_pred)
        gradient = -2 * (y_true - y_pred) / N
        
        # This gradient is passed to the backward method of the last layer
        return gradient