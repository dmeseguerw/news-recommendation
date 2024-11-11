import torch
import torch.nn as nn
import torch.nn.functional as F

class AttLayer2(nn.Module):
    """Soft alignment attention implementation in PyTorch.
    
    Attributes:
        dim (int): Attention hidden dimension.
    """

    def __init__(self, dim=200, seed=0):
        """Initialize the attention layer.
        
        Args:
            dim (int): Attention hidden dimension.
        """
        super(AttLayer2, self).__init__()
        self.dim = dim
        torch.manual_seed(seed)

        # Define weights
        self.W = nn.Parameter(torch.empty(1, dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.q = nn.Parameter(torch.empty(dim, 1))

        # Initialize weights
        nn.init.xavier_uniform_(self.W) # Xavier initialization equivalent to Glorot uniform in Keras
        nn.init.xavier_uniform_(self.q)

    def forward(self, inputs, mask=None):
        """Core implementation of soft attention.
        
        Args:
            inputs (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            mask (Tensor, optional): Mask tensor of shape (batch_size, seq_len), with 0s for masked positions.
        
        Returns:
            Tensor: Weighted sum of input tensors (batch_size, input_dim).
        """
        # Shape (batch_size, seq_len, dim)
        attention = torch.tanh(inputs @ self.W + self.b)
        
        # Shape (batch_size, seq_len, 1)
        attention = attention @ self.q

        # Shape (batch_size, seq_len)
        attention = attention.squeeze(-1)

        if mask is not None:
            # Apply mask, set masked positions to large negative value for stable softmax
            attention = attention.masked_fill(mask == 0, -1e10)
        
        # Softmax over the sequence dimension
        attention_weights = F.softmax(attention, dim=-1)
        attention_weights = attention_weights.unsqueeze(-1)

        # Weighted sum of inputs
        weighted_input = inputs * attention_weights
        return weighted_input.sum(dim=1)

    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor
        
        Args:
            input_shape (tuple): Shape of input tensor.
        
        Returns:
            tuple: Shape of output tensor.
        """
        return input_shape[0], input_shape[-1]


class SelfAttention(nn.Module):
    """Multi-head self-attention implementation in PyTorch.

    Args:
        multiheads (int): The number of heads.
        head_dim (int): Dimension of each head.
        mask_right (bool): Whether to mask right words.

    Returns:
        Tensor: Weighted sum after attention.
    """

    def __init__(self, multiheads, head_dim, seed=0, mask_right=False):
        """Initialization steps for SelfAttention.
        
        Args:
            multiheads (int): The number of heads.
            head_dim (int): Dimension of each head.
            mask_right (bool): Whether to mask right words.
        """
        super(SelfAttention, self).__init__()
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        torch.manual_seed(seed)

        # Define weights
        self.WQ = nn.Parameter(torch.empty(self.output_dim, head_dim))
        self.WK = nn.Parameter(torch.empty(self.output_dim, head_dim))
        self.WV = nn.Parameter(torch.empty(self.output_dim, head_dim))

        # Initialize weights
        nn.init.xavier_uniform_(self.WQ)
        nn.init.xavier_uniform_(self.WK)
        nn.init.xavier_uniform_(self.WV)

    def mask_attention(self, attention, seq_len, mask_mode="add"):
        """Applies masking to the attention scores if necessary.
        
        Args:
            attention (Tensor): The attention scores.
            seq_len (Tensor): The sequence lengths.
            mask_mode (str): The mode of mask, either "add" or "mul".
        
        Returns:
            Tensor: Masked attention scores.
        """
        if seq_len is None:
            return attention
        else:
            batch_size, seq_len_q, seq_len_k = attention.size()
            mask = torch.triu(torch.ones(seq_len_q, seq_len_k), diagonal=1).to(attention.device)
            if mask_mode == "mul":
                return attention * mask.unsqueeze(0)
            elif mask_mode == "add":
                return attention - (1 - mask.unsqueeze(0)) * 1e12

    def forward(self, QKV, mask=None):
        """Core logic of multi-head self-attention.

        Args:
            QKV (list): List of tensors [Q, K, V] for query, key, and value.
            mask (Tensor, optional): Mask tensor.
        
        Returns:
            Tensor: Output tensor after self-attention.
        """
        Q, K, V = QKV  # Assume Q, K, V are each (batch_size, seq_len, input_dim)

        # Linear transformations for Q, K, V
        Q = Q @ self.WQ
        K = K @ self.WK
        V = V @ self.WV

        # Reshape for multi-head attention
        Q = Q.view(Q.size(0), Q.size(1), self.multiheads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(K.size(0), K.size(1), self.multiheads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(V.size(0), V.size(1), self.multiheads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)

        # Apply mask if specified
        if self.mask_right:
            mask = torch.tril(torch.ones(attention_scores.size(-2), attention_scores.size(-1))).to(attention_scores.device)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_probs = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to V
        context = torch.matmul(attention_probs, V)

        # Concatenate heads and reshape output
        context = context.permute(0, 2, 1, 3).contiguous().view(context.size(0), -1, self.output_dim)
        
        return context
    

class ComputeMasking(nn.Module):
    """Compute if inputs contain zero values.

    Returns:
        Tensor: Float tensor where 1.0 represents non-zero values and 0.0 represents zero values.
    """

    def __init__(self):
        super(ComputeMasking, self).__init__()

    def forward(self, inputs):
        # Check if inputs are not equal to zero and cast the boolean mask to float
        mask = (inputs != 0).float()
        return mask


class OverwriteMasking(nn.Module):
    """Set values at specific positions to zero based on a mask tensor.

    Args:
        inputs (tuple or list): A pair containing the value tensor and the mask tensor.

    Returns:
        Tensor: The value tensor with masked positions set to zero.
    """

    def __init__(self):
        super(OverwriteMasking, self).__init__()

    def forward(self, inputs):
        # Expand the mask tensor along a new dimension to match the shape of the value tensor
        values, mask = inputs
        mask = mask.unsqueeze(-1)  # Adds a new dimension if needed
        return values * mask


class PersonalizedAttentivePooling(nn.Module):
    """Soft alignment attention implement.

    Args:
        dim1 (int): First dimension of value shape.
        dim2 (int): Second dimension of value shape.
        dim3 (int): Shape of query.

    Returns:
        Tensor: Weighted summary of the input value tensor.
    """
    def __init__(self, dim1, dim2, dim3, seed=0):
        super(PersonalizedAttentivePooling, self).__init__()

        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.seed = seed

        # Dropout layer
        self.dropout = nn.Dropout(0.2)

        # Dense layer (Linear layer) with Xavier uniform initialization
        self.dense = nn.Linear(dim2, dim3)
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, vecs_input, query_input):
        # Apply dropout to the input vectors
        user_vecs = self.dropout(vecs_input)

        # Apply dense layer followed by tanh activation
        user_att = torch.tanh(self.dense(user_vecs))

        # Compute attention scores: Dot product between query_input and user_att
        user_att2 = torch.matmul(query_input, user_att.transpose(-1, -2))

        # Apply softmax to the attention scores
        user_att2 = F.softmax(user_att2, dim=-1)

        # Compute the weighted sum of the values based on the attention scores
        user_vec = torch.matmul(user_att2, user_vecs)

        return user_vec
