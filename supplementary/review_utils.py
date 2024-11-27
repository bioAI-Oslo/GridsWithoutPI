import torch

import jax.numpy as jnp
from jax import jit
import numpy as np
import spatial_maps as sm

class LinearDecoder(torch.nn.Module):
    """
    Linear decoder network for SpaceNet models that decodes the spatial representation into Cartesian coordinates.
    """

    def __init__(self, n_in, n_out=2, **kwargs):
        """ Dense linear network decoder

        Parameters
        ----------
        n_in: int
            Number of inputs features.
        n_out: int
            Number of output features. Defaults to 2 (Cartesian coordinates).
        """
        super(LinearDecoder, self).__init__(**kwargs)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_in+2, n_out),
        )
        self.mse = torch.nn.MSELoss()

    def forward(self, x):
        return self.decoder(x)

    def loss_fn(self, x, y):
        return self.mse(self(x), y)

    def train_step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.loss_fn(x, y)
        loss.backward()
        optimizer.step()
        return loss.item()
    

def decode(g, decoder, r0):
    r_pred = torch.zeros((g.shape[0], g.shape[1], 2))
    r_pred[:,0] = r0
    for i in range(g.shape[1]-1):
        r_pred[:,i+1] = decoder(torch.cat((g[:,i+1], r_pred[:,i]),dim=-1))
    
    return r_pred

def band_score(ratemap, box_width):
    """
    Compute the band score, adapted from Redman et al. (2024)
    https://github.com/william-redman/Not-So-Griddy
    """
    X = np.linspace(0, box_width, ratemap.shape[0])
    Y = np.linspace(0, box_width, ratemap.shape[1])
    k = np.arange(0, 2*np.pi, 0.1)
    corrs = []
    acorr = sm.autocorrelation(ratemap, mode = "same")

    for ii in range(len(k)):
        for jj in range(len(k)):
            Z = np.outer(np.exp(1j * 2 * np.pi * k[ii] * X) , np.exp(1j * 2 * np.pi * k[jj] * Y))
            corrs.append(np.corrcoef(np.real(Z).flatten(), acorr.flatten())[0, 1])

    band_score = np.nanmax(corrs)
    return band_score

