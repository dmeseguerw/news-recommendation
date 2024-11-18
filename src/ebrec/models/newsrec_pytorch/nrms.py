# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from ebrec.models.newsrec.layers import AttLayer2, SelfAttention
#import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class NRMSModel:
    """NRMS model(Neural News Recommendation with Multi-Head Self-Attention)

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
    """

    def __init__(
        self,
        hparams: dict,
        word2vec_embedding: np.ndarray = None,
        word_emb_dim: int = 300,
        vocab_size: int = 32000,
        seed: int = None,
    ):
        """Initialization steps for NRMS."""
        super(NRMSModel, self).__init__()
        self.hparams = hparams
        self.seed = seed

        # SET SEED:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # INIT THE WORD-EMBEDDINGS:
        if word2vec_embedding is None:
            self.word2vec_embedding = nn.Embedding(vocab_size, word_emb_dim)
        else:
            self.word2vec_embedding = nn.Embedding.from_pretrained(torch.tensor(word2vec_embedding, dtype=torch.float), freeze=False)

         # Build model (the User and News Encoders)
        self.news_encoder = self._build_newsencoder()
        self.user_encoder = self._build_userencoder(self.news_encoder)

        # Loss function and optimizer setup
        self.loss_fn = self._get_loss(hparams['loss'])
        self.optimizer = self._get_opt(hparams['optimizer'], hparams['learning_rate'])

    def _get_loss(self, loss: str):
        """Make loss function, consists of data loss and regularization loss
        Returns:
            object: Loss function or loss function name
        """
        if loss == "cross_entropy_loss":
            return nn.CrossEntropyLoss()
        elif loss == "log_loss":
            return nn.BCELoss()
        else:
            raise ValueError(f"this loss not defined {loss}")
    

    def _get_opt(self, optimizer: str, lr: float):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            object: An optimizer.
        """
        # TODO: shouldn't be a string input you should just set the optimizer, to avoid stuff like this:
        # => 'WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.'
        if optimizer == "adam":
            return optim.Adam(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer type not defined: {optimizer}")
    
    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of NRMS.

        Args:
            titleencoder (object): the news encoder of NRMS.

        Return:
            object: the user encoder of NRMS.
        """
        class UserEncoder(nn.Module):
            def __init__(self, titleencoder, hparams):
                super(UserEncoder, self).__init__()
                self.titleencoder = titleencoder
                self.self_attention = SelfAttention(hparams['head_num'], hparams['head_dim'], seed=hparams['seed'])
                self.att_layer = AttLayer2(hparams['attention_hidden_dim'], seed=hparams['seed'])

            def forward(self, x):
                click_title_presents = torch.stack([self.titleencoder(y) for y in x], dim=1)
                y = self.self_attention(click_title_presents)
                user_present = self.att_layer(y)
                return user_present
        return UserEncoder(self.hparams, titleencoder, self.seed)    
        # User encoder will use the same `titleencoder` as news encoder
        
    def _build_newsencoder(self):
        class NewsEncoder(nn.Module):
            def __init__(self, word2vec_embedding, hparams):
                super(NewsEncoder, self).__init__()
                self.embedding_layer = word2vec_embedding
                self.self_attention = SelfAttention(hparams['head_num'], hparams['head_dim'], seed=hparams['seed'])
                self.dropout = nn.Dropout(hparams['dropout'])
                self.att_layer = AttLayer2(hparams['attention_hidden_dim'], seed=hparams['seed'])

            def forward(self, x):
                embedded = self.embedding_layer(x)
                y = self.dropout(embedded)
                y = self.self_attention([y, y, y])
                y = self.dropout(y)
                pred_title = self.att_layer(y)
                return pred_title

        return NewsEncoder(self.word2vec_embedding, self.hparams)
    
    
    def forward(self, his_input_title, pred_input_title, pred_input_title_one):
        user_present = self.user_encoder(his_input_title)
        
        news_present = torch.stack(
            [self.news_encoder(title) for title in pred_input_title], dim=1
        )
        preds = torch.matmul(news_present, user_present.unsqueeze(-1)).squeeze(-1)
        preds = torch.softmax(preds, dim=-1)
        
        news_present_one = self.news_encoder(pred_input_title_one.squeeze(1))
        pred_one = torch.matmul(news_present_one, user_present.unsqueeze(-1)).squeeze(-1)
        pred_one = torch.sigmoid(pred_one)
        
        return preds, pred_one    

def build_nrms(hparams):
    """
    Builds the NRMS model with separate training and scoring models.
    
    Args:
        hparams: Hyperparameters for the model (must include vocab_size, embedding_dim, filters, user_embedding_dim, etc.)
        
    Returns:
        model: The NRMS model for training and prediction.
        scorer: A scorer model for single news item scoring.
    """
    # Instantiate the main NRMS model
    model = NRMSModel(hparams)
    
    # Define example inputs to illustrate usage
    # (These would typically come from your data pipeline)
    his_input_title = torch.randint(0, hparams.vocab_size, (hparams.batch_size, hparams.history_size, hparams.title_size))
    pred_input_title = torch.randint(0, hparams.vocab_size, (hparams.batch_size, hparams.npratio + 1, hparams.title_size))
    pred_input_title_one = torch.randint(0, hparams.vocab_size, (hparams.batch_size, 1, hparams.title_size))
    
    # Forward pass through the model
    preds, pred_one = model(his_input_title, pred_input_title, pred_input_title_one)
    
    return model, preds, pred_one
