import torch 
import numpy as np
import pickle
import os

class FFGC(torch.nn.Module):
    def __init__(self, ng=256, alpha = 0.88, sigma = 1, norm = "l1"):
        super().__init__()
        self.ng = ng
        self.alpha = alpha
        self.sigma = sigma
        self.norm = norm
        # self.beta = torch.nn.Parameter(torch.tensor(1.0), requires_grad = True)
        self.beta = 1.0

        self.rg  = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, ng))
        self.relu = torch.nn.ReLU()

        self.similarity_loss_history = []
        self.capacity_loss_history = []
        self.CI_loss_history = []
        self.total_loss_history = []

    @property
    def device(self):
        return next(self.parameters()).device

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
    
    def CI_loss(self, g, r):

        # Draw distances dr from gaussian
        self.CI_scale = 0.1
        ds = self.CI_scale*abs(torch.randn(r.shape[0], device = self.device))
        # Draw random angles
        theta1 = 2*np.pi*torch.rand(r.shape[0], device = self.device)
        # Draw random angles 2
        theta2 = 2*np.pi*torch.rand(r.shape[0], device = self.device)
        # Vectors 1
        dr1 = ds[...,None]*torch.stack((torch.cos(theta1), torch.sin(theta1)), dim = -1)
        r1 = r + dr1
        # Vectors 2
        dr2 = ds[...,None]*torch.stack((torch.cos(theta2), torch.sin(theta2)), dim = -1)
        r2 = r + dr2
        # States 1
        g1 = self(r1)
        # States 2
        g2 = self(r2)
        # S1
        s1 = torch.sum((g - g1) ** 2, axis=-1) / torch.maximum(ds**2, torch.tensor(1e-14).to(self.device))
        # S2
        s2 = torch.sum((g - g2) ** 2, axis=-1) / torch.maximum(ds**2, torch.tensor(1e-14).to(self.device))
        # conf_iso_loss = (s1 - s2) ** 2
        self.scale = self.beta*0.1*256*0.0142/1000
        conf_iso_loss = (s1*ds**2-ds**2*self.scale) ** 2 + (s2*ds**2-ds**2*self.scale) ** 2
        return torch.mean(conf_iso_loss)

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
        # similarity_loss = self.alpha*self.similarity_loss(gs, labels)
        CI_loss = self.alpha*self.CI_loss(gs, labels)
        capacity_loss = (1-self.alpha)*self.capacity_loss(gs)
        loss = CI_loss + capacity_loss
        loss.backward()
        optimizer.step()
        # self.similarity_loss_history.append(similarity_loss.item())
        self.CI_loss_history.append(CI_loss.item())
        self.capacity_loss_history.append(capacity_loss.item())
        self.total_loss_history.append(loss.item())
        return loss

    def name(self):
        return f"{self.__class__.__name__}_{len(self.total_loss_history)}"

    def save(self, path=None):
        path = f"./saved-models/{self.name()}.pkl" if path is None else path
        device = self.device
        self.to(torch.device("cpu"))
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self.to(device)

    def get_model_list(self):
        model_list = ['./saved-models/'+f for f in os.listdir("./saved-models/") if f.startswith(self.__class__.__name__)]
        sorted_list = sorted(model_list, key = lambda x: int(x.split("_")[-1].split(".")[0]))
        return sorted_list

    def load(self, path=None):
        if path is None:
            model_list = self.get_model_list()
            path = model_list[-1]
        return pickle.loads(open(path, "rb").read())

    def forward_ratemaps(self, layer, output_unit=None, sort_idxs=None, rmin=-2*np.pi, rmax=2*np.pi, res=64, verbose=True):
        # define domain
        x = np.linspace(rmin, rmax, res)
        y = np.linspace(rmin, rmax, res)
        xx, yy = np.meshgrid(x, y)
        r = np.stack((xx,yy), axis = -1).reshape(-1, 2) # (res*res, 2)
        r = torch.tensor(r.astype('float32'), device = self.device)
        # investigate codomain
        if layer == 'full':
            activity = self(r)
        elif layer > -1:
            print(self.rg[:layer+1]) if verbose else None # show function composition
            activity = self.rg[:layer+1](r)
        else:
            # domain is the codomain
            activity = r
        activity = activity.detach().cpu().numpy()
        # investigate output unit 
        if output_unit is not None and layer != 'full' and layer < len(self.rg) - 1:
            weight = self.rg[layer+1].weight.detach().cpu().numpy()[output_unit] # (ncells,)
            print("Pattern formation of output unit", output_unit, "in layer", layer+1) if verbose else None
            activity = activity * weight # (res*res, ncells)
        # sort by aggregate activity
        #sort_idxs = np.argsort(np.sum(activity, axis = 0))[::-1] if sort_idxs is None else sort_idxs
        sort_idxs = np.argsort(np.linalg.norm(activity, axis = 0))[::-1] if sort_idxs is None else sort_idxs
        activity = activity[:,sort_idxs]
        return activity.T.reshape(-1, res, res), sort_idxs

    
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
    