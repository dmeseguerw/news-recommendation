import torch
import torch.nn as nn
from ebrec.models.newsrec_pytorch.layers import AttLayer2, SelfAttention

class NewsEncoder(nn.Module):
    def __init__(self, word2vec_embedding, hparams_nrms, seed):
        super(NewsEncoder, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(word2vec_embedding, freeze=False)
        self.dropout = nn.Dropout(hparams_nrms.dropout)
        # self.self_attention = SelfAttention(hparams_nrms.head_num, hparams_nrms.head_dim, seed)
        self.multihead_attention = nn.MultiheadAttention(hparams_nrms.head_dim, hparams_nrms.head_num)
        self.attention = AttLayer2(hparams_nrms.attention_hidden_dim, seed)
        
    def forward(self, sequences_input_title):
        embedded_sequences = self.embedding(sequences_input_title)
        y = self.dropout(embedded_sequences)
        y = self.multihead_attention(y)
        y = self.dropout(y)
        pred_title = self.attention(y)
        return pred_title