import torch
import torch.nn as nn
from ebrec.models.newsrec_pytorch.layers import AttLayer2, SelfAttention
from ebrec.models.newsrec_pytorch.model_config import hparams_nrms

class NewsEncoder(nn.Module):
    def __init__(self, word2vec_embedding, hparams_nrms, seed):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word2vec_embedding, freeze=False)
        self.dropout = nn.Dropout(hparams_nrms.dropout)
        self.self_attention = SelfAttention(hparams_nrms.head_num, hparams_nrms.head_dim, seed)
        self.attention = AttLayer2(hparams_nrms.attention_hidden_dim, seed)
        
    def forward(self, sequences_input_title):
        embedded_sequences = self.embedding(sequences_input_title)
        y = self.dropout(embedded_sequences)
        y = self.self_attention(y)
        y = self.dropout(y)
        pred_title = self.attention(y)
        return pred_title

class UserEncoder(nn.Module):
    def __init__(self, title_encoder, hparams_nrms, seed):
        super(UserEncoder, self).__init__()
        self.title_encoder = title_encoder
        self.self_attention = SelfAttention(hparams_nrms.head_num, hparams_nrms.head_dim, seed)
        self.attention = AttLayer2(hparams_nrms.attention_hidden_dim, seed)
        
    def forward(self, his_input_title):
        batch_size, history_size, title_size = his_input_title.shape
        
        # Reshape for titleencoder: treat each news title independently
        his_input_title_flat = his_input_title.view(-1, title_size)  # Shape: (batch_size * history_size, title_size)
        click_title_presents = self.titleencoder(his_input_title_flat)  # Shape: (batch_size * history_size, hidden_dim)
        
        # Reshape back to include history_size
        click_title_presents = click_title_presents.view(batch_size, history_size, -1)  # Shape: (batch_size, history_size, hidden_dim)

        # Self-attention over the historical clicked news representations
        y = self.self_attention(click_title_presents)  # Shape: (batch_size, history_size, hidden_dim)
        
        # Dropout
        y = self.dropout(y)
        
        # Attention layer for user representation
        user_present = self.attention(y)  # Shape: (batch_size, hidden_dim)
        
        return user_present
        