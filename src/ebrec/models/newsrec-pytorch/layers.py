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


class SelfAttention(layers.Layer):
    """Multi-head self attention implement.

    Args:
        multiheads (int): The number of heads.
        head_dim (object): Dimention of each head.
        mask_right (boolean): whether to mask right words.

    Returns:
        object: Weighted sum after attention.
    """

    def __init__(self, multiheads, head_dim, seed=0, mask_right=False, **kwargs):
        """Initialization steps for AttLayer2.

        Args:
            multiheads (int): The number of heads.
            head_dim (object): Dimention of each head.
            mask_right (boolean): whether to mask right words.
        """

        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        self.seed = seed
        super(SelfAttention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """Compute shape of output tensor.

        Returns:
            tuple: output shape tuple.
        """

        return (input_shape[0][0], input_shape[0][1], self.output_dim)

    def build(self, input_shape):
        """Initialization for variables in SelfAttention.
        There are three variables in SelfAttention, i.e. WQ, WK ans WV.
        WQ is used for linear transformation of query.
        WK is used for linear transformation of key.
        WV is used for linear transformation of value.

        Args:
            input_shape (object): shape of input tensor.
        """

        self.WQ = self.add_weight(
            name="WQ",
            shape=(int(input_shape[0][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.WK = self.add_weight(
            name="WK",
            shape=(int(input_shape[1][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        self.WV = self.add_weight(
            name="WV",
            shape=(int(input_shape[2][-1]), self.output_dim),
            initializer=keras.initializers.glorot_uniform(seed=self.seed),
            trainable=True,
        )
        super(SelfAttention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode="add"):
        """Mask operation used in multi-head self attention

        Args:
            seq_len (object): sequence length of inputs.
            mode (str): mode of mask.

        Returns:
            object: tensors after masking.
        """

        if seq_len is None:
            return inputs
        else:
            mask = K.one_hot(indices=seq_len[:, 0], num_classes=K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, axis=1)

            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)

            if mode == "mul":
                return inputs * mask
            elif mode == "add":
                return inputs - (1 - mask) * 1e12

    def call(self, QKVs):
        """Core logic of multi-head self attention.

        Args:
            QKVs (list): inputs of multi-head self attention i.e. qeury, key and value.

        Returns:
            object: ouput tensors.
        """
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs
            Q_len, V_len = None, None
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(
            Q_seq, shape=(-1, K.shape(Q_seq)[1], self.multiheads, self.head_dim)
        )
        Q_seq = K.permute_dimensions(Q_seq, pattern=(0, 2, 1, 3))

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(
            K_seq, shape=(-1, K.shape(K_seq)[1], self.multiheads, self.head_dim)
        )
        K_seq = K.permute_dimensions(K_seq, pattern=(0, 2, 1, 3))

        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(
            V_seq, shape=(-1, K.shape(V_seq)[1], self.multiheads, self.head_dim)
        )
        V_seq = K.permute_dimensions(V_seq, pattern=(0, 2, 1, 3))
        A = tf.matmul(Q_seq, K_seq, adjoint_a=False, adjoint_b=True) / K.sqrt(
            K.cast(self.head_dim, dtype="float32")
        )

        A = K.permute_dimensions(
            A, pattern=(0, 3, 2, 1)
        )  # A.shape=[batch_size,K_sequence_length,Q_sequence_length,self.multiheads]

        A = self.Mask(A, V_len, "add")
        A = K.permute_dimensions(A, pattern=(0, 3, 2, 1))

        if self.mask_right:
            ones = K.ones_like(A[:1, :1])
            lower_triangular = K.tf.matrix_band_part(ones, num_lower=-1, num_upper=0)
            mask = (ones - lower_triangular) * 1e12
            A = A - mask
        A = K.softmax(A)

        O_seq = tf.matmul(A, V_seq, adjoint_a=True, adjoint_b=False)
        O_seq = K.permute_dimensions(O_seq, pattern=(0, 2, 1, 3))

        O_seq = K.reshape(O_seq, shape=(-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, "mul")
        return O_seq

    def get_config(self):
        """add multiheads, multiheads and mask_right into layer config.

        Returns:
            dict: config of SelfAttention layer.
        """
        config = super(SelfAttention, self).get_config()
        config.update(
            {
                "multiheads": self.multiheads,
                "head_dim": self.head_dim,
                "mask_right": self.mask_right,
            }
        )
        return config


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
