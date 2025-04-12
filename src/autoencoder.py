from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.init as init


class Dictionary(ABC):
    dict_size : int # number of features in the dictionary
    activation_dim : int # dimension of the activation vectors

    @abstractmethod
    def encode(self, x):
        pass
    
    @abstractmethod
    def decode(self, f):
        pass
    
    
class GatedAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with separate gating and magnitude networks.
    """
    def __init__(self, activation_dim, dict_size, initialization='default', device=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.device = device
        self.decoder_bias = nn.Parameter(torch.empty(activation_dim, device=device))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=False, device=device)
        self.r_mag = nn.Parameter(torch.empty(dict_size, device=device))
        self.gate_bias = nn.Parameter(torch.empty(dict_size, device=device))
        self.mag_bias = nn.Parameter(torch.empty(dict_size, device=device))
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False, device=device)
        if initialization == 'default':
            self._reset_parameters()
        else:
            initialization(self)

    def _reset_parameters(self):
        """
        Default method for initializing GatedSAE weights.
        """
        # biases are initialized to zero
        init.zeros_(self.decoder_bias)
        init.zeros_(self.r_mag)
        init.zeros_(self.gate_bias)
        init.zeros_(self.mag_bias)

        # decoder weights are initialized to random unit vectors
        dec_weight = torch.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self, x, return_gate=False):
        """
        Returns features, gate value (pre-Heavyside)
        """
        x = x.to(self.device)
        x_enc = self.encoder(x.to(self.decoder_bias.device) - self.decoder_bias)

        # gating network
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).float()

        # magnitude network
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = nn.ReLU()(pi_mag)

        f = f_gate * f_mag
        
        if return_gate:
            return f, nn.ReLU()(pi_gate)

        return f

    def decode(self, f):
        return self.decoder(f) + self.decoder_bias
    
    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)

        f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if output_features:
            return x_hat, f
        else:
            return x_hat

    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = torch.load(path)
        dict_size, activation_dim = state_dict['encoder.weight'].shape
        autoencoder = GatedAutoEncoder(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder
