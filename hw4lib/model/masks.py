import torch

''' 
TODO: Implement this function.

Specification:
- Function should create a padding mask that identifies padded positions in the input
- Mask should be a boolean tensor of shape (N, T) where:
  * N = batch size from padded_input
  * T = sequence length from padded_input
- True values indicate padding positions that should be masked
- False values indicate valid positions that should not be masked
- Padding is assumed to be on the right side of sequences
- Each sequence in the batch may have different valid lengths
- Mask should be on same device as input tensor
'''
def PadMask(padded_input, input_lengths):
    """ 
    Create a mask to identify non-padding positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: The actual lengths of each sequence before padding, shape (N,).
    Returns:
        A boolean mask tensor with shape (N, T), where: 
            - padding positions are marked with True 
            - non-padding positions are marked with False.
    """
    # TODO: Implement PadMask
    
    # Get batch size and sequence length
    N = padded_input.shape[0]
    T = padded_input.shape[1]
    
    # Create a range tensor [0, 1, 2, ..., T-1] of shape (T,)
    # This represents the position indices
    positions = torch.arange(T, device=padded_input.device)
    
    # Expand dimensions to enable broadcasting
    # positions: (T,) -> (1, T)
    # input_lengths: (N,) -> (N, 1)
    positions = positions.unsqueeze(0)  # (1, T)
    input_lengths = input_lengths.unsqueeze(1)  # (N, 1)
    
    # Broadcasting comparison: (N, 1) with (1, T) -> (N, T)
    # For each position, check if it's >= the valid length
    # True means it's a padding position (should be masked)
    # False means it's a valid position (should not be masked)
    pad_mask = positions >= input_lengths
    
    return pad_mask


''' 
TODO: Implement this function.

Specification:
- Function should create a causal mask for self-attention
- Mask should be a boolean tensor of shape (T, T) where T is sequence length
- True values indicate positions that should not attend to each other
- False values indicate positions that can attend to each other
- Causal means each position can only attend to itself and previous positions
- Mask should be on same device as input tensor
- Mask should be upper triangular (excluding diagonal)
'''
def CausalMask(padded_input):
    """ 
    Create a mask to identify non-causal positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
    
    Returns:
        A boolean mask tensor with shape (T, T), where: 
            - non-causal positions (don't attend to) are marked with True 
            - causal positions (can attend to) are marked with False.
    """
    # TODO: Implement CausalMask
    
    # Get sequence length
    T = padded_input.shape[1]
    
    # Create a lower triangular matrix (including diagonal) of ones
    # torch.tril creates a lower triangular matrix with 1s below and on diagonal, 0s above
    # We want the opposite: 0s (False) for causal positions, 1s (True) for non-causal
    # So we use torch.triu with diagonal offset of 1 to get upper triangular (excluding diagonal)
    causal_mask = torch.triu(torch.ones(T, T, device=padded_input.device), diagonal=1).bool()
    
    # Alternative approach (equivalent):
    # causal_mask = ~torch.tril(torch.ones(T, T, device=padded_input.device)).bool()
    
    return causal_mask