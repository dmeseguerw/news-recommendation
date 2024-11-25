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

word2vec_embedding = torch.randn((10000, 300))
seed = 42
news_encoder = NewsEncoder(word2vec_embedding, hparams_nrms, seed)

sequences_input_title = torch.randint(0, 10000, (16, hparams_nrms.title_size))
output = news_encoder(sequences_input_title)