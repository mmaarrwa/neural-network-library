import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        # Stores the learning rate (η) centrally
        self.learning_rate = learning_rate

    def step(self, parameter, gradient):
        """
        Updates the parameter (W or b) using the SGD rule: 
        P_new = P_old - η * (∂L/∂P)
        
        Args:
            parameter (np.array): The weight or bias array (P_old).
            gradient (np.array): The calculated gradient for that parameter (∂L/∂P).
        """
        # Apply the update rule: W_new = W_old - η * gradient
        parameter -= self.learning_rate * gradient