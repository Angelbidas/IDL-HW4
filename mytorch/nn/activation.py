import numpy as np
class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        Z_max = np.max(Z, axis=self.dim, keepdims=True)  # For numerical stability
        exp_Z = np.exp(Z - Z_max)  # Subtract max for stability
        self.A = exp_Z / np.sum(exp_Z, axis=self.dim, keepdims=True)
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = shape[self.dim]
           
        # Reshape input to 2D
        if len(shape) > 2:
            original_shape = shape
            self.A = self.A.reshape(-1, C)
            dLdA = dLdA.reshape(-1, C)
        # Reshape back to original dimensions if necessary
        dLdZ = np.empty_like(self.A)
        for i in range(self.A.shape[0]):  # Iterate over the batch
            softmax_row = self.A[i].reshape(-1, 1)  # (C, 1)
            jacobian = np.diagflat(softmax_row) - softmax_row @ softmax_row.T  # (C, C)
            dLdZ[i] = dLdA[i] @ jacobian
        if len(shape) > 2:
            # Restore shapes to original
            self.A = self.A.reshape(original_shape)
            dLdZ = dLdZ.reshape(original_shape)

        return dLdZ
 

    