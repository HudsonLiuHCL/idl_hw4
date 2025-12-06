import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer

     
class SelfAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer with masked self-attention and feed-forward sublayers.
    Used in the decoder-only Transformer architecture.  
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):

        super().__init__()
       
        # Initialize the sublayers      
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)  # Masked self-attention
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)  # Feed-forward network

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, 
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        x, mha_attn_weights = self.self_attn(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        
        # Apply feed-forward sublayer
        x = self.ffn(x)
        
        return x, mha_attn_weights

## -------------------------------------------------------------------------------------------------    
class CrossAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer with masked self-attention, cross-attention, and feed-forward sublayers.
    Used in the encoder-decoder Transformer architecture.
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):

        super().__init__()

        # Initialize the sublayers  
        self.self_attn  = SelfAttentionLayer(d_model, num_heads, dropout)   # Masked self-attention
        self.cross_attn = CrossAttentionLayer(d_model, num_heads, dropout)  # Cross-attention
        self.ffn        = FeedForwardLayer(d_model, d_ff, dropout)          # Feed-forward network

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, 
                dec_key_padding_mask: Optional[torch.Tensor] = None, 
                enc_key_padding_mask: Optional[torch.Tensor] = None, 
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x, self_attn_weights = self.self_attn(x, key_padding_mask=dec_key_padding_mask, attn_mask=attn_mask)
        x, cross_attn_weights = self.cross_attn(x, enc_output, key_padding_mask=enc_key_padding_mask)

        x = self.ffn(x)

        return x, self_attn_weights, cross_attn_weights
## -------------------------------------------------------------------------------------------------