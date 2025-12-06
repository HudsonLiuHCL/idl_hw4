import torch.nn as nn
import torch 
from typing import Tuple, Optional

'''
Complete implementation of transformer sublayers.
All three sublayers follow Pre-LN architecture.
'''

class SelfAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 1.
    This layer is responsible for the causally-masked self-attention mechanism.
    '''
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        '''
        Initialize the SelfAttentionLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        '''
        super().__init__()
        
        # Initialize the multi-head attention mechanism
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Expect (batch, seq, feature) format
        )
        
        # Initialize the normalization layer
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, 
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the SelfAttentionLayer.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)   
            key_padding_mask (Optional[torch.Tensor]): The padding mask. shape: (batch_size, seq_len)
            attn_mask (Optional[torch.Tensor]): The attention mask. shape: (seq_len, seq_len)

        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, d_model)
            mha_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)   
        '''
        # Store residual connection
        residual = x
        
        # Apply pre-normalization (Pre-LN architecture)
        x = self.norm(x)
        
        # Self-attention: query = key = value = x
        x, mha_attn_weights = self.mha(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True
        )
        
        # Apply dropout and add residual connection
        x = self.dropout(x)
        x = residual + x
        
        return x, mha_attn_weights

## -------------------------------------------------------------------------------------------------  
class CrossAttentionLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 2.
    This layer is responsible for the cross-attention mechanism between encoder and decoder.
    '''
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        '''
        Initialize the CrossAttentionLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        '''
        super().__init__()
        
        # Initialize the multi-head attention mechanism
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Expect (batch, seq, feature) format
        )
        
        # Initialize the normalization layer
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None, 
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the CrossAttentionLayer.
        Args:
            x (torch.Tensor): The input tensor from decoder. shape: (batch_size, seq_len_dec, d_model)   
            y (torch.Tensor): The input tensor from encoder. shape: (batch_size, seq_len_enc, d_model)
            key_padding_mask (Optional[torch.Tensor]): The padding mask for encoder. shape: (batch_size, seq_len_enc)
            attn_mask (Optional[torch.Tensor]): The attention mask. shape: (seq_len_dec, seq_len_enc)

        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len_dec, d_model)
            mha_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len_dec, seq_len_enc)   
        '''
        # Store residual connection
        residual = x
        
        # Apply pre-normalization (Pre-LN architecture)
        x = self.norm(x)

        # Cross-attention: query = x (decoder), key = value = y (encoder)
        x, mha_attn_weights = self.mha(
            query=x,
            key=y,
            value=y,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True
        )
        
        # Apply dropout and add residual connection
        x = self.dropout(x)
        x = residual + x
        
        return x, mha_attn_weights

## -------------------------------------------------------------------------------------------------  
class FeedForwardLayer(nn.Module):
    '''
    Pre-LN Decoder Sub-Layer 3.
    This layer is responsible for the position-wise feed-forward network.
    '''
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        '''
        Initialize the FeedForwardLayer. 
        Args:
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        '''
        super().__init__()

        # Initialize the feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),      # First projection: expand
            nn.GELU(),                      # GELU activation
            nn.Dropout(dropout),            # Dropout for regularization
            nn.Linear(d_ff, d_model)        # Second projection: back to d_model
        )
        
        # Initialize the normalization layer
        self.norm = nn.LayerNorm(d_model)
        
        # Initialize the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for the FeedForwardLayer.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)   

        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, d_model)
        ''' 
        # Store residual connection
        residual = x
        
        # Apply pre-normalization (Pre-LN architecture)
        x = self.norm(x)
        
        # Apply feed-forward network
        x = self.ffn(x)
        
        # Apply dropout and add residual connection
        x = self.dropout(x)
        x = residual + x
        
        return x