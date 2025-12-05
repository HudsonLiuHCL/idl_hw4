import torch
from torch import nn
import math

'''
TODO: Implement this Module.

Specification:
- Module should add positional information to input embeddings
- Uses sinusoidal position encodings as described in "Attention Is All You Need"
- Positional encoding matrix should have shape (1, max_len, d_model)
- Even indices use sine functions, odd indices use cosine functions
- Wavelengths form geometric progression from 2π to 10000·2π
- Encoding values should be on same device as input tensor
- Should handle any sequence length up to max_len
- Should raise error if input sequence length exceeds max_len
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """
        Initialize the PositionalEncoding.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        
        Steps:
        1. Call parent class constructor using super().__init__()
        2. Call create_pe_table to initialize positional encoding matrix
        """     
        super().__init__()
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        """
        Create the positional encoding table.
        
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        
        Side Effects:
            - Initializes the positional encoding buffer 'pe' 
              of shape (1, max_len, d_model) (in order to broadcast with input tensor)
        """
        # TODO: Implement create_pe_table
        
        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Create position indices: [0, 1, 2, ..., max_len-1]
        # Shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create division term for the exponential
        # div_term = 1 / (10000^(2i/d_model)) for i = 0, 1, 2, ..., d_model//2
        # This can be rewritten as: exp(-log(10000) * 2i / d_model)
        # Shape: (d_model // 2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices (0, 2, 4, ...)
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (1, 3, 5, ...)
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (max_len, d_model) -> (1, max_len, d_model)
        # This allows broadcasting with input of shape (B, T, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer to save with model state
        # Buffers are tensors that should be saved but not trained
        self.register_buffer('pe', pe)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEncoding.
        Args:
            x (torch.Tensor): The input tensor of shape (B x T x d_model)
        Returns:
            torch.Tensor: Input with positional encoding added (B x T x d_model)
        Errors:
            - ValueError: If sequence length exceeds maximum length 
        """
        # TODO: Implement forward
        # Step 1: Get sequence length from input tensor
        seq_len = x.size(1)
        
        # Step 2: Verify sequence length doesn't exceed maximum length, raise error if it does
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds the maximum length {self.pe.size(1)}")
        
        # Step 3: Add positional encodings to input
        # pe shape: (1, max_len, d_model)
        # We slice to get only the encodings we need: (1, seq_len, d_model)
        # Broadcasting: (B, T, d_model) + (1, T, d_model) -> (B, T, d_model)
        x = x + self.pe[:, :seq_len, :]
        
        return x