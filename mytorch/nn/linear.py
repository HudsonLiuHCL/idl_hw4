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
        
        # Store the original input shape
        self.input_shape = A.shape
        
        # Get the number of input features (last dimension)
        in_features = self.input_shape[-1]
        
        # Flatten all batch dimensions: (*, in_features) -> (batch_size, in_features)
        # where batch_size = product of all dimensions except the last
        batch_size = np.prod(self.input_shape[:-1])
        A_reshaped = A.reshape(batch_size, in_features)
        
        # Perform linear transformation: Z = A @ W^T + b
        # A_reshaped: (batch_size, in_features)
        # W^T: (in_features, out_features)
        # b: (out_features,)
        Z_reshaped = A_reshaped @ self.W.T + self.b
        
        # Reshape back to original batch dimensions + out_features
        output_shape = list(self.input_shape[:-1]) + [self.W.shape[0]]
        Z = Z_reshaped.reshape(output_shape)
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass
        
        # Get dimensions
        out_features = self.W.shape[0]
        in_features = self.W.shape[1]
        
        # Flatten the gradient to 2D: (*, out_features) -> (batch_size, out_features)
        batch_size = np.prod(self.input_shape[:-1])
        dLdZ_reshaped = dLdZ.reshape(batch_size, out_features)
        
        # Flatten the input to 2D: (*, in_features) -> (batch_size, in_features)
        A_reshaped = self.A.reshape(batch_size, in_features)
        
        # Compute gradients (refer to the equations in the writeup)
        
        # dL/dA = dL/dZ @ W
        # dLdZ_reshaped: (batch_size, out_features)
        # W: (out_features, in_features)
        # Result: (batch_size, in_features)
        dLdA_reshaped = dLdZ_reshaped @ self.W
        
        # dL/dW = (dL/dZ)^T @ A
        # dLdZ_reshaped^T: (out_features, batch_size)
        # A_reshaped: (batch_size, in_features)
        # Result: (out_features, in_features)
        self.dLdW = dLdZ_reshaped.T @ A_reshaped
        
        # dL/db = sum over batch dimension of dL/dZ
        # dLdZ_reshaped: (batch_size, out_features)
        # Result: (out_features,)
        self.dLdb = np.sum(dLdZ_reshaped, axis=0)
        
        # Reshape dLdA back to original input shape
        self.dLdA = dLdA_reshaped.reshape(self.input_shape)
        
        # Return gradient of loss wrt input
        return self.dLdA