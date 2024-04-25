import torch
import torch.utils
import xarray as xr
import numpy as np
import copy

class TaxiDataset(torch.utils.data.Dataset):
    def __init__(self, data, indices=None, seed=None, no_norm=False):
        self.np_random = np.random.default_rng(seed)

        # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'

        if indices is not None:
            self.data = data.isel(sample=indices)
        else:
            self.data = data

        self.norm = not no_norm

        triu_rows, triu_cols = np.triu_indices(self.data.cost.shape[1], k=1)
        cost = self.data.cost.values[:, triu_rows, triu_cols] # select half of cost matrix because it is symmetric
        capacity = self.data.capacity.values
        demand = self.data.demand.values
        latents = np.hstack((capacity, demand, cost))
        if self.norm:
            latents = (latents - latents.mean(axis=0))/latents.std(axis=0)
        self.latents = torch.Tensor(latents).float()

        obs = self.data.allocation.values
        if self.norm:
            obs = (obs - obs.mean(axis=0))/obs.std(axis=0)
        self.obs = torch.Tensor(obs).float()

        capacity_perturbation = self.data.capacity_perturbation.values
        demand_perturbation = self.data.demand_perturbation.values
        action = np.hstack((capacity_perturbation, demand_perturbation))
        self.action = torch.Tensor(action).float()

        _, action, _, _, latents = self.__getitem__(0)
        self.gt_gc = torch.Tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        ]).T.float()
        assert self.gt_gc.shape[0] == latents.shape[1], f'gt_gc is hardcoded and the number of rows ({self.gt_gc.shape[0]}) does not match the dataset ({latents.shape[1]})'
        assert self.gt_gc.shape[1] == action.shape[0], f'gt_gc is hardcoded and the number of columns ({self.gt_gc.shape[1]}) does not match the dataset ({action.shape[0]})'

    def __len__(self):
        return len(self.data.coords['sample'])

    def __getitem__(self, index):
        obs = self.obs[index]

        action = self.action[index]

        discrete_action_placeholder = torch.zeros(1).long()

        valid = True

        latents = self.latents[index]

        return obs.view(1, -1), action, discrete_action_placeholder, valid, latents.view(1, -1)
