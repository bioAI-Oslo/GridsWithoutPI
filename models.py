import torch 
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

class FFGC(torch.nn.Module):
    def __init__(self, ng=256, alpha = 0.225, sigma = 1.8, rho = 1, norm = "l1"):
        super().__init__()
        self.ng = ng
        self.alpha = alpha
        self.sigma = sigma
        self.rho = torch.nn.Parameter(torch.tensor(rho, dtype=torch.float32), requires_grad = False)
        self.norm = norm

        self.rg  = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, ng))
        self.relu = torch.nn.ReLU()

        self.distance_loss_history = []
        self.capacity_loss_history = []
        self.CI_loss_history = []
        self.total_loss_history = []
        self.rhos = []

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

    def plot_envelope(self, fig=None, ax=None, ab=2*np.pi, res=64, **kwargs):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        mesh = np.linspace(-ab, ab, res)
        xx, yy = np.meshgrid(mesh, mesh)
        r = np.stack((xx,yy), axis = -1).reshape(-1, 2)
        r = torch.tensor(r.astype('float32'), device = self.device)
        r0 = r[res**2//2 + res//2]
        #envelope = torch.exp(-torch.sum((r - r0)**2, axis = 1)/(2*(1.5*self.sigma)**2)).detach().cpu().numpy()
        envelope = torch.distributions.normal.Normal(0, self.sigma).log_prob(torch.linalg.norm(r - r0, dim = 1)).detach().cpu().numpy()
        envelope = np.exp(envelope)
        im = ax.imshow(envelope.reshape(res, res), **kwargs)
        return fig, ax, im

    def distance_loss(self, g, r):
        # reshape to accomodate FF and RNN
        g = torch.reshape(g, (-1, g.shape[-1])) 
        r = torch.reshape(r, (-1, r.shape[-1]))
        dg = torch.nn.functional.pdist(g) # state distance
        dr = torch.nn.functional.pdist(r) # spatial distance
        # loss envelope function
        envelope = torch.exp(-dr**2/(2*self.sigma**2))
        diff = (dg - dr)**2
        return torch.mean(diff*envelope)

    def capacity_loss(self, g):
        # reshape to accomodate FF and RNN
        g = torch.reshape(g, (-1, g.shape[-1])) 
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

        distance_loss = self.alpha*self.distance_loss(gs, labels)
        capacity_loss = (1-self.alpha)*self.capacity_loss(gs)
        loss = distance_loss + capacity_loss

        loss.backward()
        optimizer.step()
        
        # log losses 
        self.distance_loss_history.append(distance_loss.item())
        self.capacity_loss_history.append(capacity_loss.item())
        self.total_loss_history.append(loss.item())
        self.rhos.append(self.rho.item())
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
        self.vg = torch.nn.Linear(2, self.ng, bias = False)

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
    
    # def jacobi_CI_loss(self, r):
    #     m = self.metric_tensor(r)
    #     loss = torch.mean((self.rho*m - torch.eye(m.shape[-1], device = m.device))**2)
    #     return loss

    # def metric_tensor(self, r):
    #     # Batched Jacobian
    #     J = torch.vmap(torch.func.jacfwd(self.forward))(r)#.requires_grad_())
    #     # Batched J.T @ J
    #     m = torch.matmul(J.permute(0, 2, 1), J)
    #     return m

    # def distance_loss(self, g, r):
    #     # reshape to accomodate FF and RNN
    #     g = torch.reshape(g, (-1, g.shape[-1])) 
    #     r = torch.reshape(r, (-1, r.shape[-1]))
    #     perturbed_r = r + np.sqrt(self.sigma)*torch.randn(r.shape, device = self.device)
    #     perturbed_g = self(perturbed_r)
    #     dr = torch.sum((r - perturbed_r)**2, axis = 1) # spatial distance
    #     dg = torch.sum((g - perturbed_g)**2, axis = 1) # state distance
    #     # envelope = torch.distributions.normal.Normal(0, self.sigma).log_prob(dr)
    #     #envelope = torch.exp(envelope) / torch.exp(torch.distributions.normal.Normal(0, self.sigma).log_prob(torch.tensor(0.)))
    #     envelope = torch.exp(-dr**2/(2*(1.5*self.sigma)**2))
    #     diff = envelope*(dg - self.rho*dr)**2
    #     return torch.mean(diff)
