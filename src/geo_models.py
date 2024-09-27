from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import r2_score, auroc


__all__ = [
    "BaseGEOModel", "AutoEncoder", "VariationalAutoEncoder",
    "ConstraintAutoEncoder", "MultiTask", "AttentionMultiTask"
]


def _build_linear(layer_dims: List[int]):
    layers = list()
    for i, (_in, _out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        layers.append(nn.Linear(_in, _out))
        if i < len(layer_dims)-2:
            #layers.append(nn.BatchNorm1d(_out))
            layers.append(nn.ReLU())
    return layers


class AttentionBlock(nn.Module):
    # Inspired by Attention is all you need, but without final add&norm

    def __init__(self, _in: int, _out: int, num_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=_in, num_heads=num_heads)
        self.bn = nn.BatchNorm1d(_in)
        self.ffn = nn.Sequential(nn.Linear(_in, _in), nn.ReLU(), nn.Linear(_in, _out))

    def forward(self, x):
        a, _ = self.mha(x, x, x, need_weights=False)
        y = self.bn(a+x)
        return self.ffn(y)


def _build_attention(layer_dims: List[int], num_heads: int = 8):
    layers = list()

    # create embedding layer
    _in, _out = layer_dims[:2]
    layers.append(nn.Linear(_in, _out))
    
    # create attention blocks for the rest of the layers
    for _in, _out in zip(layer_dims[1:-1], layer_dims[2:]):
        layers.append(AttentionBlock(_in, _out, num_heads))

    return layers


class BaseGEOModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        raise NotImplementedError()
    
    def criterion(self, predicted, x, y, w):
        raise NotImplementedError()
    
    def alt_score(self, predicted, x, y, w):
        raise NotImplementedError()
    
    def alt_score_name(self):
        raise NotImplementedError()


class AutoEncoder(BaseGEOModel):

    def __init__(self, input_dim: int = 884, encoder_layers: List[int] = [256],
                 latent_dim: int = 128, decoder_layers: List[int] = None):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = input_dim

        # If decoder_layers are not provided, reversed encoder_layers are used
        self.encoder_layer_dims = encoder_layers
        if decoder_layers is not None:
            self.decoder_layer_dims = decoder_layers
        else:
            self.decoder_layer_dims= encoder_layers[::-1]
        
        self._encoder_layers = _build_linear([self.input_dim]+self.encoder_layer_dims+[self.latent_dim])
        self.encoder = nn.Sequential(*self._encoder_layers)
        self._decoder_layers = _build_linear([self.latent_dim]+self.decoder_layer_dims+[self.output_dim])
        self.decoder = nn.Sequential(*self._decoder_layers)

        #self._critertion = nn.MSELoss(reduction="mean")

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def criterion(self, predicted, x, y, w):
        #return self._critertion(predicted, x)
        return (predicted-x).pow(2).sum(axis=1).mean() # mean MSE per sample

    def alt_score(self, pred, x, y, w):
        return r2_score(pred, x)

    def alt_score_name(self):
        return "r2"


class ConstraintAutoEncoder(AutoEncoder):

    def __init__(self, input_dim: int = 884, encoder_layers: List[int] = [256],
                 latent_dim: int = 128, decoder_layers: List[int] = None, beta: float = 1.0):
        super().__init__(input_dim, encoder_layers, latent_dim, decoder_layers)
        self.beta = beta

    def criterion(self, predicted, x, y, w):
        rec_loss = super().criterion(predicted, x, y, w)
        z = self.encoder(x)
        enc_loss = z.pow(2).sum(axis=1).mean()
        return rec_loss + self.beta*enc_loss


class VariationalAutoEncoder(AutoEncoder):

    def __init__(self, input_dim: int = 884, encoder_layers: List[int] = [256],
                 latent_dim: int = 128, decoder_layers: List[int] = None, beta: float = 0.5):
        super().__init__(input_dim, encoder_layers, latent_dim, decoder_layers)
        del self.encoder
        self.beta = beta

        self._common_encoder = nn.Sequential(*self._encoder_layers[:-1])
        self._mean_model = self._encoder_layers[-1]
        self.encoder = nn.Sequential(self._common_encoder, self._mean_model)
        self._variance_model = nn.Linear(self.encoder_layer_dims[-1], self.latent_dim)

    def forward(self, x):
        # Reparametrization trick from
        # https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py#L107
        mean, logvar = self.mean_logvar(x)
        std = torch.exp(0.5 * logvar)
        z = torch.randn_like(std)
        return self.decoder(mean+z*std)

    def mean_logvar(self, x):
        c = self._common_encoder(x)
        return self._mean_model(c), self._variance_model(c)
    
    # TODO create vae loss
    def criterion(self, predicted, x, y, w):
        # https://discuss.pytorch.org/t/correct-implementation-of-vae-loss/146750
        rec_loss = super().criterion(predicted, x, y, w)
        mu, logvar = self.mean_logvar(x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        return rec_loss + self.beta*kld_loss


class MultiTask(BaseGEOModel):

    def __init__(self, input_dim: int = 884, encoder_layers: List[int] = [256],
                 latent_dim: int = 128, num_tasks: int = 695):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = num_tasks

        self.encoder_layer_dims = encoder_layers

        self._encoder_layers = _build_linear([self.input_dim]+self.encoder_layer_dims+[self.latent_dim])
        self.encoder = nn.Sequential(*self._encoder_layers)
        self.prediction_head = nn.Sequential(nn.ReLU(), nn.Linear(self.latent_dim, self.output_dim))

        #self._critertion = nn.BCELoss(reduction="mean")

    def forward(self, x):
        l = self.encoder(x)
        z = self.prediction_head(l)
        return torch.sigmoid(z)

    def criterion(self, pred, x, y, w):
        #where = (w > 0)
        #return self._critertion(pred[where], y[where])
        loss = F.binary_cross_entropy(pred, y, weight=w, reduction="sum")
        return loss / x.shape[0]

    def alt_score(self, pred, x, y, w):
        where = (w > 0)
        return auroc(pred[where], y[where].type(torch.int), task="binary")
    
    def alt_score_name(self):
        return "auc"


class AttentionMultiTask(MultiTask):

    def __init__(self, input_dim: int = 884, encoder_layers: List[int] = [256],
                 latent_dim: int = 128, num_tasks: int = 695, embedding_dim: int = 880, num_heads: int = 8):
        super().__init__(input_dim, encoder_layers, latent_dim, num_tasks)
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(f"'embedding_dim' ({embedding_dim}) must be divisible by 'num_heads' ({num_heads})")
        for el in self.encoder_layer_dims:
            if el % self.num_heads != 0:
                raise ValueError(f"'encoder_layers' ({self.encoder_layer_dims}) contains a value that is not divisible by 'num_heads' ({num_heads})")

        self._encoder_layers = _build_attention([self.input_dim, self.embedding_dim]+self.encoder_layer_dims+[self.latent_dim], num_heads)
        self.encoder = nn.Sequential(*self._encoder_layers)
