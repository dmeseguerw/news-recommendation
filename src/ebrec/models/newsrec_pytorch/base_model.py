'''This file contains the base_model.py original code commented out along with the new pytorch equivalent implementation 
and notes on the changes.'''

from typing import Any, Dict
#from tensorflow import keras
#import tensorflow as tf
import numpy as np
import abc
#---NEW CODE---
import torch
import torch.nn as nn
import torch.optim as optim
#--------------


__all__ = ["BaseModel"]


#class BaseModel:
#    """Basic class of models

#    Attributes:
#        hparams (object): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
#        graph (object): An optional graph.
#        seed (int): Random seed.
#    """
# --- NEW CODE ---
class BaseModel(abc.ABC):
    """Basic class of models in PyTorch.

    Attributes:
        hparams (dict): A dictionary containing hyperparameters.
        seed (int): Random seed.
    """
#--------------

    def __init__(
        self,
        hparams: Dict[str, Any],
        word2vec_embedding: np.ndarray = None,
        # if 'word2vec_embedding' not provided:
        word_emb_dim: int = 300,
        vocab_size: int = 32000,
        seed=None,
    ):
#        """Initializing the model. Create common logics which are needed by all deeprec models, such as loss function,
#        parameter set.

#        Args:
#            hparams (object): Hold the entire set of hyperparameters.
#            seed (int): Random seed.
#        """
        """Initialize the model. Set up loss function, optimizer, and embeddings.

        Args:
            hparams (dict): Hyperparameters dictionary.
            seed (int): Random seed.
        """
        self.seed = seed
#        tf.random.set_seed(seed)
#        np.random.seed(seed)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # ASSIGN 'hparams':
        self.hparams = hparams

        # INIT THE WORD-EMBEDDINGS:
        if word2vec_embedding is None:
            #self.word2vec_embedding = np.random.rand(vocab_size, word_emb_dim)
            self.word2vec_embedding = torch.tensor(
                np.random.rand(vocab_size, word_emb_dim), dtype=torch.float32
            )
        else:
            #self.word2vec_embedding = word2vec_embedding
            self.word2vec_embedding = torch.tensor(word2vec_embedding, dtype=torch.float32)

#        # BUILD AND COMPILE MODEL:
        # Build model, compile() is a tensorflow step
        self.model, self.scorer = self._build_graph()
        #self.loss = self._get_loss(self.hparams.loss)
        self.loss = self._get_loss(self.hparams["loss"])
        self.train_optimizer = self._get_opt(
            #optimizer=self.hparams.optimizer, lr=self.hparams.learning_rate
            optimizer=self.hparams["optimizer"], lr=self.hparams["learning_rate"]
        )
        #self.model.compile(loss=self.loss, optimizer=self.train_optimizer)

    @abc.abstractmethod
    def _build_graph(self):
        """Subclass will implement this."""
        pass

    def _get_loss(self, loss: str):
#        """Make loss function, consists of data loss and regularization loss
#
#        Returns:
#            object: Loss function or loss function name
#        """
        """Define the loss function.

        Returns:
            object: PyTorch loss function.
        """
        if loss == "cross_entropy_loss":
            #data_loss = "categorical_crossentropy"
            data_loss = nn.CrossEntropyLoss()
        elif loss == "log_loss":
            #data_loss = "binary_crossentropy"
            data_loss = nn.BCELoss() # Comment by Laura: why if input is log_loss binary is returned ??
        else:
            raise ValueError(f"this loss not defined {loss}")
        return data_loss

    def _get_opt(self, optimizer: str, lr: float):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            object: An optimizer.
        """
        if optimizer == "adam":
            #train_opt = keras.optimizers.Adam(learning_rate=lr)
            train_opt = optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"this optimizer not defined {optimizer}")

        return train_opt
