import torch 
import numpy as np
import pickle

class FFGC(torch.nn.Module):
    def __init__(self, ng=256, alpha = 0.9, sigma = 1, norm = "l1"):
        super().__init__()
        self.ng = ng
        self.alpha = alpha
        self.sigma = sigma
        self.norm = norm

        self.rg  = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, ng))
        self.relu = torch.nn.ReLU()

        self.similarity_loss_history = []
        self.capacity_loss_history = []
        self.total_loss_history = []

    def norm_relu(self, x):
        rx = self.relu(x)
        norm = (torch.linalg.norm(rx, dim = -1)[...,None])
        return rx/torch.maximum(norm, 1e-13*torch.ones(norm.shape, device = x.device))

    def forward(self, r):
        g = self.rg(r)
        return self.norm_relu(g)

    def similarity_loss(self, g, r):
        # reshape to accomodate FF and RNN
        g = torch.reshape(g, (-1, g.shape[-1])) 
        r = torch.reshape(r, (-1, r.shape[-1]))
        dg = torch.nn.functional.pdist(g) # state distance
        dr = torch.nn.functional.pdist(r) # spatial distance
        # gaussian similarity difference
        diff = torch.exp(-dr**2/(2*self.sigma**2)) - torch.exp(-dg**2)
        # loss envelope function
        envelope = torch.exp(-dr**2/(2*(1.5*self.sigma)**2))
        return torch.mean(diff**2*envelope)

    def capacity_loss(self, g):
        # reshape to accomodate FF and RNN
        g = torch.reshape(g, (-1, g.shape[-1])) ###############
        if self.norm == "l1":
            return -torch.mean(g) # g is non-negative
        elif self.norm == "l2":
            return -torch.mean(torch.mean(g, dim = 0)**2)
        else:
            raise ValueError
    
    def loss_minima(self):
        return self.alpha-1

    def train_step(self, inputs, labels, optimizer):
        optimizer.zero_grad()
        gs = self(inputs)
        similarity_loss = self.alpha*self.similarity_loss(gs, labels)
        capacity_loss = (1-self.alpha)*self.capacity_loss(gs)
        loss = similarity_loss + capacity_loss
        loss.backward()
        optimizer.step()
        self.similarity_loss_history.append(similarity_loss.item())
        self.capacity_loss_history.append(capacity_loss.item())
        self.total_loss_history.append(loss.item())
        return loss

    def save(self, path=None):
        path = f"./saved-models/{self.__class__.__name__}.pkl" if path is None else path
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path=None):
        path = f"./saved-models/{self.__class__.__name__}.pkl" if path is None else path
        return pickle.loads(open(path, "rb").read())

    
class RNNGC(FFGC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.gg = torch.nn.Linear(self.ng, self.ng, bias = False)
        torch.nn.init.eye_(self.gg.weight)

        self.vg = torch.nn.Linear(2, self.ng)
        self.relu = torch.nn.ReLU()

    def recurrent_step(self, g_prev, v):
        h = self.gg(g_prev) + self.vg(v)
        return self.norm_relu(h)

    def forward(self, inputs):
        # inputs = (initial position, velocities)
        r0, v = inputs
        g = [self.norm_relu(self.rg(r0))] # initial state

        # RNN 
        for i in range(v.shape[1]):
            g.append(self.recurrent_step(g[-1], v[:,i]))
        return torch.stack(g, dim = 1)
    