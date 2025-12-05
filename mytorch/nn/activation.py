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
        
        # Numerical stability: subtract max along the dimension
        Z_shifted = Z - np.max(Z, axis=self.dim, keepdims=True)
        
        # Compute exponentials
        exp_Z = np.exp(Z_shifted)
        
        # Compute softmax
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
        
        # Store original shape for later restoration
        original_shape = shape
        
        # Move the softmax dimension to the last position
        # This makes the computation easier to handle
        self.A = np.moveaxis(self.A, self.dim, -1)
        dLdA = np.moveaxis(dLdA, self.dim, -1)
        
        # Reshape input to 2D
        if len(shape) > 2:
            # Flatten all dimensions except the last one (which is now the softmax dimension)
            batch_size = np.prod(self.A.shape[:-1])
            self.A = self.A.reshape(batch_size, C)
            dLdA = dLdA.reshape(batch_size, C)
        
        # Now self.A and dLdA are 2D: (batch_size, C)
        # Compute gradient using the Jacobian approach
        batch_size = self.A.shape[0]
        dLdZ = np.zeros_like(self.A)
        
        for i in range(batch_size):
            # Get the softmax probabilities for this sample
            a = self.A[i]  # Shape: (C,)
            
            # Compute the Jacobian matrix for this sample
            # J[m,n] = a[m] * (δ[m,n] - a[n])
            # where δ[m,n] is the Kronecker delta
            jacobian = np.diag(a) - np.outer(a, a)
            
            # Multiply gradient by Jacobian
            dLdZ[i] = dLdA[i] @ jacobian.T
        
        # Reshape back to original dimensions if necessary
        if len(original_shape) > 2:
            # First reshape to the moved-axis shape (before moveaxis)
            moved_shape = list(original_shape)
            moved_shape.pop(self.dim)
            moved_shape.append(C)
            
            self.A = self.A.reshape(moved_shape)
            dLdZ = dLdZ.reshape(moved_shape)
        
        # Move the axis back to its original position
        self.A = np.moveaxis(self.A, -1, self.dim)
        dLdZ = np.moveaxis(dLdZ, -1, self.dim)
        
        return dLdZ