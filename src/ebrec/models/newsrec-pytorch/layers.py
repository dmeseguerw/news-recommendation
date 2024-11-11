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
    

class ComputeMasking(layers.Layer):
    """Compute if inputs contains zero value.

    Returns:
        bool tensor: True for values not equal to zero.
    """

    def __init__(self, **kwargs):
        super(ComputeMasking, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        mask = K.not_equal(inputs, 0)
        return K.cast(mask, K.floatx())

    def compute_output_shape(self, input_shape):
        return input_shape


class OverwriteMasking(layers.Layer):
    """Set values at spasific positions to zero.

    Args:
        inputs (list): value tensor and mask tensor.

    Returns:
        object: tensor after setting values to zero.
    """

    def __init__(self, **kwargs):
        super(OverwriteMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        super(OverwriteMasking, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs[0] * K.expand_dims(inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def PersonalizedAttentivePooling(dim1, dim2, dim3, seed=0):
    """Soft alignment attention implement.
    Attributes:
        dim1 (int): first dimention of value shape.
        dim2 (int): second dimention of value shape.
        dim3 (int): shape of query

    Returns:
        object: weighted summary of inputs value.
    """
    vecs_input = keras.Input(shape=(dim1, dim2), dtype="float32")
    query_input = keras.Input(shape=(dim3,), dtype="float32")

    user_vecs = layers.Dropout(0.2)(vecs_input)
    user_att = layers.Dense(
        dim3,
        activation="tanh",
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed),
        bias_initializer=keras.initializers.Zeros(),
    )(user_vecs)
    user_att2 = layers.Dot(axes=-1)([query_input, user_att])
    user_att2 = layers.Activation("softmax")(user_att2)
    user_vec = layers.Dot((1, 1))([user_vecs, user_att2])

    model = keras.Model([vecs_input, query_input], user_vec)
    return model
