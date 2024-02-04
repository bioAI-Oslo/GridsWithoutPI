import torch
import numpy as np

class DatasetMaker(object):
    # Simple dataset maker; square box + bounce off walls
    def __init__(self, box_size = 2*np.pi, von_mises_scale = 4*np.pi, rayleigh_scale = 0.15):
        self.box_size = box_size # box_size x box_size enviroment
        self.von_mises_scale = von_mises_scale
        self.rayleigh_scale = rayleigh_scale

    def bounce(self, r, v):
        # bounce off walls if next step lands outside
        outside = np.abs(r + v) >= self.box_size
        v[outside] = -v[outside]
        return v

    def generate_data(self, samples, timesteps, device = "cpu"):
        r = np.zeros((samples, timesteps, 2)) # positions
        s = np.random.rayleigh(self.rayleigh_scale, (samples, timesteps)) # speeds

        # initial conditions
        prev_hd = np.random.uniform(0, 2*np.pi, samples) # previous head direction
        r[:,0] = np.random.uniform(-self.box_size, self.box_size, (samples, 2))

        for i in range(timesteps - 1):
            hd = np.random.vonmises(prev_hd, self.von_mises_scale, samples)
            prop_v = s[:,i,None]*np.stack((np.cos(hd), np.sin(hd)),axis=-1)
            v = self.bounce(r[:,i], prop_v)
            prev_hd = np.arctan2(v[:,1], v[:,0])
            r[:,i+1] = r[:,i] + v

        v = np.diff(r, axis = 1) # velocities

        return torch.tensor(r.astype('float32'), device = device), torch.tensor(v.astype('float32'), device = device)