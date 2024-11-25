import torch
import torch.nn as nn
from ebrec.models.newsrec_pytorch.news_encoder import NewsEncoder
from ebrec.models.newsrec_pytorch.user_encoder import UserEncoder

class NRMSModel(nn.Module):
    def __init__(self, hparams_nrms, word2vec_embedding, seed):
        super(NRMSModel, self).__init__()
        self.news_encoder = NewsEncoder(word2vec_embedding, hparams_nrms, seed)
        self.user_encoder = UserEncoder(self.news_encoder, hparams_nrms, seed)
        print(self.news_encoder)
        print(self.user_encoder)
        self.hparams_nrms = hparams_nrms
    
    def forward(self, his_input_title, pred_input_title):
        news_present = torch.stack([self.news_encoder(title) for title in pred_input_title], dim=1)
        user_present = self.user_encoder(his_input_title)
        preds = torch.matmul(news_present, user_present.unsqueeze(-1)).squeeze(-1)
        preds = torch.softmax(preds, dim=-1)
        return preds