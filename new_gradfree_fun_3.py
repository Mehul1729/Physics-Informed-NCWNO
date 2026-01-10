import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class allen_cahn_gradient_free(nn.Module):
    def __init__(self, n_neighbour=4, radius=0.020, dt=0.01, nu=0.0001):
        super().__init__()
        self.loss_function = nn.MSELoss(reduction='sum')
        self.n_neighbour = n_neighbour
        self.radius = radius
        self.dt = dt
        self.nu = nu

    def neighbour_index(self, x_f):
       
        N_f = x_f.shape[0]
        
        dists = torch.cdist(x_f, x_f) # Shape [N, N]
        
        dists.fill_diagonal_(float('inf'))
        
        
        
        mask = dists > self.radius
        dists_masked = dists.clone()
        dists_masked[mask] = float('inf')
        
        # Get nearest neighbours
        # We get the indices of the smallest distances
        _, sorted_indices = torch.topk(dists_masked, k=self.n_neighbour, dim=1, largest=False)
        
        
        
        zn = sorted_indices
    
        chosen_dists = torch.gather(dists_masked, 1, zn)
        is_invalid = chosen_dists == float('inf')
        
        if is_invalid.any():
            self_indices = torch.arange(N_f, device=x_f.device).unsqueeze(1).expand_as(zn)
            zn = torch.where(is_invalid, self_indices, zn)
            
        return zn

    def inverse_index(self, x_f):
        N_f = x_f.shape[0]
        zn = self.neighbour_index(x_f) # [N, k]

        neighbors = x_f[zn].squeeze(-1) # [N, k]
        center = x_f.squeeze(-1).unsqueeze(1) # [N, 1]
        
        # dx: [N, k]
        dx = neighbors - center

        # Vectorized dot product:
        cov = torch.einsum('nk,nk->n', dx, dx).reshape(N_f, 1, 1)
        
        # Add identity noise
        cov += 1e-8 * torch.eye(1, device=x_f.device).unsqueeze(0)
        
        # Batch Inverse
        invs = torch.linalg.inv(cov) # [N, 1, 1]
        
        return invs, zn

    def loss(self, up_batch, usol_batch, u_t_batch, x_f, u_t1_batch, zn, invs):
        
        loss_u = self.loss_BC(up_batch, usol_batch)
        loss_f = self.loss_PDE(u_t_batch, x_f, u_t1_batch, zn, invs)

        return 10 * loss_u + loss_f






# -----------------------------------------------------------------------






class nagumo_gradient_free(allen_cahn_gradient_free):
    
    def __init__(self, n_neighbour=6, radius=0.024, dt=0.01, nu=0.08, alpha=-0.5):
        super().__init__(n_neighbour, radius, dt, nu)
        self.alpha = alpha

    def loss_PDE(self, u_t_batch, x_f, u_t1_batch, zn, invs):
        u_x_t_batch = self.grad1(x_f, u_t_batch, zn, invs)
        u_x_t1_batch = self.grad1(x_f, u_t1_batch, zn, invs)
        u_xx_t_batch = self.grad2(x_f, u_x_t_batch, zn, invs)
        u_xx_t1_batch = self.grad2(x_f, u_x_t1_batch, zn, invs)

        f_t_batch = self.nu * u_xx_t_batch + u_t_batch * (1 - u_t_batch) * (u_t_batch - self.alpha)
        f_t1_batch = self.nu * u_xx_t1_batch + u_t1_batch * (1 - u_t1_batch) * (u_t1_batch - self.alpha)

        res_batch = (u_t1_batch - u_t_batch) - 0.5 * self.dt * (f_t_batch + f_t1_batch)
        return self.loss_function(res_batch, torch.zeros_like(res_batch))


class burger_gradient_free(allen_cahn_gradient_free):
    
    def __init__(self, n_neighbour=5, radius=0.020, dt=0.02, nu=0.1): # reduced the number of neighbors to make it more spatially localised and also reduced the radius from 0.024 to 0.020
        super().__init__(n_neighbour, radius, dt, nu)

    def loss_PDE(self, u_t_batch, x_f, u_t1_batch, zn, invs):
        u_x_t_batch = self.grad1(x_f, u_t_batch, zn, invs)
        u_xx_t_batch = self.grad2(x_f, u_x_t_batch, zn, invs)
        u_x_t1_batch = self.grad1(x_f, u_t1_batch, zn, invs)
        u_xx_t1_batch = self.grad2(x_f, u_x_t1_batch, zn, invs)

        f_t_batch = self.nu * u_xx_t_batch - u_t_batch * u_x_t_batch
        f_t1_batch = self.nu * u_xx_t1_batch - u_t1_batch * u_x_t1_batch
 
        # debug : replacing f_t by f_t in cn loss final term
        res_batch = (u_t1_batch - u_t_batch) - 0.5 * self.dt * (f_t_batch + f_t1_batch)
        return self.loss_function(res_batch, torch.zeros_like(res_batch))


class wave_gradient_free(allen_cahn_gradient_free):
    
    def __init__(self, n_neighbour=8, radius=0.024, dt=0.01, c=1):
        super().__init__(n_neighbour, radius, dt)
        self.c = c

    def loss_PDE(self, u_t_batch, x_f, u_t1_batch, zn, invs):
        
        u_xx_t_batch = self.grad2(x_f, self.grad1(x_f, u_t_batch, zn, invs), zn, invs)
        u_xx_t1_batch = self.grad2(x_f, self.grad1(x_f, u_t1_batch, zn, invs), zn, invs)

        f_t_batch = self.c**2 * u_xx_t_batch
        f_t1_batch = self.c**2 * u_xx_t1_batch


        res_batch = (u_t1_batch - u_t_batch) - 0.5 * self.dt * (f_t_batch + f_t1_batch)
        return self.loss_function(res_batch, torch.zeros_like(res_batch))


class advection_gradient_free(allen_cahn_gradient_free):
  
    def __init__(self, n_neighbour=9, radius=0.024, dt=0.01, c=1): # test run : changed radius fro,m 0.024 to 0.010
        super().__init__(n_neighbour, radius, dt)
        self.c = c

    def loss_PDE(self, u_t_batch, x_f, u_t1_batch, zn, invs):
        u_x_t_batch = self.grad1(x_f, u_t_batch, zn, invs)
        u_x_t1_batch = self.grad1(x_f, u_t1_batch, zn, invs)

        f_t_batch = -self.c * u_x_t_batch
        f_t1_batch = -self.c * u_x_t1_batch

        res_batch = (u_t1_batch - u_t_batch) - 0.5 * self.dt * (f_t_batch + f_t1_batch)
        return self.loss_function(res_batch, torch.zeros_like(res_batch))


class heat_gradient_free(allen_cahn_gradient_free):

    def __init__(self, n_neighbour=6, radius=0.024, dt=0.01, alpha=0.01):
        super().__init__(n_neighbour, radius, dt)
        self.alpha = alpha

    def loss_PDE(self, u_t_batch, x_f, u_t1_batch, zn, invs):
        u_xx_t_batch = self.grad2(x_f, self.grad1(x_f, u_t_batch, zn, invs), zn, invs)
        u_xx_t1_batch = self.grad2(x_f, self.grad1(x_f, u_t1_batch, zn, invs), zn, invs)

        f_t_batch = self.alpha * u_xx_t_batch
        f_t1_batch = self.alpha * u_xx_t1_batch

        res_batch = (u_t1_batch - u_t_batch) - 0.5 * self.dt * (f_t_batch + f_t1_batch)
        return self.loss_function(res_batch, torch.zeros_like(res_batch))