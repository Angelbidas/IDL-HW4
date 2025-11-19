import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)

    
    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
      
        
        # Store input for backward pass
        self.A = A
        Z = self.A @ self.W.T + self.b
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass
        # Compute gradients (refer to the equations in the writeup)
        batch_shape = self.A.shape[:-1]  # All dimensions except the last one
        in_features = self.A.shape[-1]
        out_features = self.W.shape[0]
        
        # Reshape inputs to 2D for easier computation
        A_reshaped = self.A.reshape(-1, in_features)  # (batch_size, in_features)
        dLdZ_reshaped = dLdZ.reshape(-1, out_features)  # (batch_size, out_features)
        
        # Compute gradients
        self.dLdW = dLdZ_reshaped.T @ A_reshaped  # (out_features, in_features)
        self.dLdb = np.sum(dLdZ_reshaped, axis=0)  # (out_features,)
        dLdA_reshaped = dLdZ_reshaped @ self.W  # (batch_size, in_features)
        
        # Reshape dLdA back to original input shape
        dLdA = dLdA_reshaped.reshape(*batch_shape, in_features)
        
        return dLdA

