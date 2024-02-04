import torch 
import numpy as np

class FFGC(torch.nn.Module):
    def __init__(self, ng, device, alpha = 0.5, sigma = 1):
        super().__init__()
        self.ng = ng
        self.alpha = alpha
        self.sigma = sigma

        self.rg  = torch.nn.Sequential(
            torch.nn.Linear(2, 64, device = device),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128, device = device),
            torch.nn.ReLU(),
            torch.nn.Linear(128, ng, device = device))
        
        self.relu = torch.nn.ReLU()

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
        # g = torch.reshape(g, (-1, g.shape[-1])) ###############
        # mean_vector = torch.mean(g, dim = (0, 1))

        mean_vector = torch.mean(g, dim = (0, 1))
        return -torch.mean(mean_vector**2) #####################
        # return -torch.mean(g) #####################

    def train_step(self, inputs, labels, optimizer):
        optimizer.zero_grad()
        gs = self(inputs)
        loss = self.alpha*self.similarity_loss(gs, labels) + (1-self.alpha)*self.capacity_loss(gs)
        loss.backward()
        optimizer.step()
        return loss
    
class RNNGC(FFGC):
    def __init__(self, ng, device, alpha = 0.5, sigma = 1):
        super().__init__(ng, device, alpha, sigma)
        
        # initial state encoder
        self.rg  = torch.nn.Sequential(
            torch.nn.Linear(2, 64, device = device),
            torch.nn.ReLU(),
            torch.nn.Linear(64, ng, device = device))
        
        self.gg = torch.nn.Linear(ng, ng, device = device, bias = False)
        torch.nn.init.eye_(self.gg.weight)

        self.vg = torch.nn.Linear(2, ng, device = device, bias = False)
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
    