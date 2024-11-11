# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from ebrec.models.newsrec.layers import AttLayer2, SelfAttention
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class NRMSModel_docvec:
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
        seed: int = None,
        newsencoder_units_per_layer: list[int] = [512, 512, 512],
    ):
        """Initialization steps for NRMS."""
        self.hparams = hparams
        self.seed = seed
        self.newsencoder_units_per_layer = newsencoder_units_per_layer

        # SET SEED:
        torch.manual_seed(seed)
        np.random.seed(seed)
        # BUILD AND COMPILE MODEL:
        self.model, self.scorer = self._build_graph()
        data_loss = self._get_loss(self.hparams.loss)
        train_optimizer = self._get_opt(
            optimizer=self.hparams.optimizer, lr=self.hparams.learning_rate
        )
        self.model.compile(loss=data_loss, optimizer=train_optimizer)

    def _get_loss(self, loss: str):
        """Make loss function, consists of data loss and regularization loss
        Returns:
            object: Loss function or loss function name
        """
        if loss == "cross_entropy_loss":
            data_loss = "categorical_crossentropy"
        elif loss == "log_loss":
            data_loss = "binary_crossentropy"
        else:
            raise ValueError(f"this loss not defined {loss}")
        return data_loss

    def _get_opt(self, optimizer: str, lr: float):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            object: An optimizer.
        """
        if optimizer == "adam":
            train_opt = optim.Adam(self.parameters(), lr=lr)
        else:
            raise ValueError(f"this optimizer not defined {optimizer}")
        return train_opt

    def _build_graph(self):
        """Build NRMS model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_userencoder(self, titleencoder):
        """The main function to create user encoder of NRMS.

        Args:
            titleencoder (object): the news encoder of NRMS.

        Return:
            object: the user encoder of NRMS.
        """
        class UserEncoder(nn.Module):
            def __init__(self, hparams, titleencoder, seed):
                super(UserEncoder, self).__init__()
                self.hparams = hparams
                self.titleencoder = titleencoder
                self.self_attention = SelfAttention(hparams.head_num, hparams.head_dim, seed)
                self.att_layer = AttLayer2(hparams.attention_hidden_dim, seed)

            def forward(self, his_input_title):
                batch_size, history_size, title_size = his_input_title.size()
                his_input_title = his_input_title.view(batch_size * history_size, title_size)
                click_title_presents = self.titleencoder(his_input_title)
                click_title_presents = click_title_presents.view(batch_size, history_size, -1)
                y = self.self_attention(click_title_presents)
                user_present = self.att_layer(y)
                return user_present

        return UserEncoder(self.hparams, titleencoder, self.seed)

    def _build_newsencoder(self, units_per_layer: list[int] = list[512, 512, 512]):
        """THIS IS OUR IMPLEMENTATION.
        The main function to create a news encoder.

        Parameters:
            units_per_layer (int): The number of neurons in each Dense layer.

        Return:
            object: the news encoder.
        """
        DOCUMENT_VECTOR_DIM = self.hparams.title_size
        OUTPUT_DIM = self.hparams.head_num * self.hparams.head_dim

        class NewsEncoder(nn.Module):
            def __init__(self, hparams, units_per_layer):
                super(NewsEncoder, self).__init__()
                layers = []
                input_dim = hparams.title_size
                for units in units_per_layer:
                    layers.append(nn.Linear(input_dim, units))
                    layers.append(nn.ReLU())
                    layers.append(nn.BatchNorm1d(units))
                    layers.append(nn.Dropout(hparams.dropout))
                    input_dim = units
                layers.append(nn.Linear(input_dim, OUTPUT_DIM))
                layers.append(nn.ReLU())
                self.model = nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)

        return NewsEncoder(self.hparams, units_per_layer)

    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        class NRMS(nn.Module):
            def __init__(self, userencoder, newsencoder):
                super(NRMS, self).__init__()
                self.userencoder = userencoder
                self.newsencoder = newsencoder

            def forward(self, his_input_title, pred_input_title):
                user_present = self.userencoder(his_input_title)
                news_present = torch.stack([self.newsencoder(title) for title in pred_input_title], dim=1)
                preds = torch.matmul(news_present, user_present.unsqueeze(-1)).squeeze(-1)
                preds = torch.softmax(preds, dim=-1)
                return preds

        class Scorer(nn.Module):
            def __init__(self, userencoder, newsencoder):
                super(Scorer, self).__init__()
                self.userencoder = userencoder
                self.newsencoder = newsencoder

            def forward(self, his_input_title, pred_input_title_one):
                user_present = self.userencoder(his_input_title)
                news_present_one = self.newsencoder(pred_input_title_one.squeeze(1))
                pred_one = torch.sigmoid(torch.matmul(news_present_one, user_present))
                return pred_one

        titleencoder = self._build_newsencoder(units_per_layer=self.newsencoder_units_per_layer)
        self.userencoder = self._build_userencoder(titleencoder)
        self.newsencoder = titleencoder

        model = NRMS(self.userencoder, self.newsencoder)
        scorer = Scorer(self.userencoder, self.newsencoder)

        return model, scorer