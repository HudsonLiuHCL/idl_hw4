import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        # Apply softmax along the last dimension (source sequence length S)
        self.softmax = Softmax(dim=-1)
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # TODO: Implement forward pass
        
        # Get the embedding dimension from the last dimension of Q
        d_k = Q.shape[-1]
        
        # Calculate attention scores: (N, ..., H, L, S)
        # (N, ..., H, L, E) @ (N, ..., H, E, S) -> (N, ..., H, L, S)
        # Need to transpose K to swap the last two dimensions
        K_T = np.swapaxes(K, -2, -1)  # (N, ..., H, S, E) -> (N, ..., H, E, S)
        scaled_dot_product = (Q @ K_T) / np.sqrt(d_k)
        
        # Apply mask before softmax if provided
        # If mask is not None, add -self.eps to the attention scores for positions to ignore
        if mask is not None:
            # Where mask is True, set to -eps (very negative number)
            # This makes softmax output ~0 for those positions
            scaled_dot_product = np.where(mask, -self.eps, scaled_dot_product)

        # Compute attention scores: Apply softmax along S dimension (N, ..., H, L, S)
        self.attention_scores = self.softmax.forward(scaled_dot_product)
        
        # Store values for backward pass
        self.Q = Q
        self.K = K
        self.V = V
        self.d_k = d_k
        self.scaled_dot_product = scaled_dot_product

        # Calculate output: (N, ..., H, L, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        output = self.attention_scores @ V

        # Return output
        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass

        # Calculate gradients for V: (N, ..., H, S, Ev)
        # Output = attention_scores @ V
        # d_output: (N, ..., H, L, Ev)
        # attention_scores: (N, ..., H, L, S)
        # We need: (N, ..., H, S, Ev)
        # Use the transpose of stored softmax output to swap last two dimensions   
        attention_scores_T = np.swapaxes(self.attention_scores, -2, -1)  # (N, ..., H, S, L)
        d_V = attention_scores_T @ d_output  # (N, ..., H, S, L) @ (N, ..., H, L, Ev) -> (N, ..., H, S, Ev)
        
        # Calculate gradients for attention scores
        # Output = attention_scores @ V
        # d_attention_scores = d_output @ V^T
        # (N, ..., H, L, Ev) @ (N, ..., H, Ev, S) -> (N, ..., H, L, S)
        V_T = np.swapaxes(self.V, -2, -1)  # (N, ..., H, S, Ev) -> (N, ..., H, Ev, S)
        d_attention_scores = d_output @ V_T  # (N, ..., H, L, Ev) @ (N, ..., H, Ev, S) -> (N, ..., H, L, S)
        
        # Backprop through softmax
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
        
        # Scale gradients by sqrt(d_k)
        d_scaled_dot_product = d_scaled_dot_product / np.sqrt(self.d_k)
        
        # Calculate gradients for Q and K
        # scaled_dot_product = (Q @ K^T) / sqrt(d_k)
        # d_Q = d_scaled_dot_product @ K
        # (N, ..., H, L, S) @ (N, ..., H, S, E) -> (N, ..., H, L, E)   
        d_Q = d_scaled_dot_product @ self.K
        
        # d_K = d_scaled_dot_product^T @ Q
        # (N, ..., H, S, L) @ (N, ..., H, L, E) -> (N, ..., H, S, E)
        d_scaled_dot_product_T = np.swapaxes(d_scaled_dot_product, -2, -1)  # (N, ..., H, L, S) -> (N, ..., H, S, L)
        d_K = d_scaled_dot_product_T @ self.Q  # (N, ..., H, S, L) @ (N, ..., H, L, E) -> (N, ..., H, S, E)
        
        # Return gradients for Q, K, V
        return d_Q, d_K, d_V