import torch
import torch.utils
import xarray as xr
import numpy as np
import copy

class TaxiDataset(torch.utils.data.Dataset):
    def __init__(self, data, indices=None, no_norm=False, include_latent_cost=False, include_offsets_in_obs=False):
        if indices is not None:
            self.data = data.isel(sample=indices)
        else:
            self.data = data

        self.norm = not no_norm

        capacity = self.data.capacity.values
        demand = self.data.demand.values
        if include_latent_cost:
            triu_rows, triu_cols = np.triu_indices(self.data.cost.shape[1], k=1)
            cost = self.data.cost.values[:, triu_rows, triu_cols] # select half of cost matrix because it is symmetric
            latents = np.hstack((capacity, demand, cost))
        else:
            latents = np.hstack((capacity, demand))
        if self.norm:
            latents = (latents - latents.mean(axis=0))/latents.std(axis=0)
        self.latents = torch.Tensor(latents).float()

        allocations = self.data.allocation.values
        if include_offsets_in_obs:
            unused_capacity = self.data.unused_capacity.values
            unserved_demand = self.data.unserved_demand.values
            obs = np.hstack((allocations.reshape(allocations.shape[0], -1), unused_capacity, unserved_demand))
        else:
            obs = allocations
        if self.norm:
            obs = (obs - obs.mean(axis=0))/obs.std(axis=0)
        self.obs = torch.Tensor(obs).float()

        capacity_perturbation = self.data.capacity_perturbation.values
        demand_perturbation = self.data.demand_perturbation.values
        action = np.hstack((capacity_perturbation, demand_perturbation))
        self.action = torch.Tensor(action).float()

        _, action, _, _, latents = self.__getitem__(0)
        if include_latent_cost:
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
        else:
            self.gt_gc = torch.eye(latents.shape[1], action.shape[0]).float()

    def __len__(self):
        return len(self.data.coords['sample'])

    def __getitem__(self, index):
        obs = self.obs[index]

        action = self.action[index]

        discrete_action_placeholder = torch.zeros(1).long()

        valid = True

        latents = self.latents[index]

        return obs.view(1, -1), action, discrete_action_placeholder, valid, latents.view(1, -1)
