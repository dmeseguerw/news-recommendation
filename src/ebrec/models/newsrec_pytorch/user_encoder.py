import torch
import torch.nn as nn
from ebrec.models.newsrec_pytorch.layers import AttLayer2, SelfAttention
from ebrec.models.newsrec_pytorch.model_config import hparams_nrms

class UserEncoder(nn.Module):
    def __init__(self, news_encoder, hparams_nrms, seed):
        super(UserEncoder, self).__init__()
        self.news_encoder = news_encoder
        self.multihead_attention = nn.MultiheadAttention(hparams_nrms.head_dim, hparams_nrms.head_num)
        self.additive_attention = AttLayer2(hparams_nrms.attention_hidden_dim, seed)
        self.dropout = nn.Dropout(hparams_nrms.dropout)
        
    def forward(self, his_input_title):
        batch_size, history_size, title_size = his_input_title.shape
        
        # Reshape for titleencoder: treat each news title independently
        his_input_title_flat = his_input_title.view(-1, title_size)  # Shape: (batch_size * history_size, title_size)
        click_title_presents = self.news_encoder(his_input_title_flat)  # Shape: (batch_size * history_size, hidden_dim)
        
        # Reshape back to include history_size
        click_title_presents = click_title_presents.view(batch_size, history_size, -1)  # Shape: (batch_size, history_size, hidden_dim)

        # Self-attention over the historical clicked news representations
        y = self.multihead_attention(click_title_presents*3)  # Shape: (batch_size, history_size, hidden_dim)
        
        # Dropout
        y = self.dropout(y)
        
        # Attention layer for user representation
        user_present = self.additive_attention(y) 
        
        return user_present
        