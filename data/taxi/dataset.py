import torch
import torch.utils
import xarray as xr
import numpy as np

class TaxiDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, seed=None, no_norm=False, gt_gc=None, c_dim=0):
        self.np_random = np.random.default_rng(seed)

        self.datapath = datapath
        self.dataset = xr.open_dataset(datapath)
        # specifying indices allows us to only select from those indices
        # if samples is not None:
        #     self.dataset = self.dataset.isel(sample=samples)

        # for some reason only the observations are normed
        self.norm = not no_norm
        if self.norm:
            mean = self.dataset.allocation.mean(dim='sample')
            std = self.dataset.allocation.std(dim='sample')
            self.dataset['allocation_norm'] = (self.dataset.allocation - mean) / std

    def __len__(self):
        return len(self.dataset.sample)

    def __getitem__(self, item):
        sample = self.dataset.isel(sample=item)

        obs = sample.allocation_norm.values if self.norm else sample.allocation.values

        action = np.hstack((sample.capacity_perturbation.values, sample.demand_perturbation.values))

        discrete_action_placeholder = torch.zeros(1).long()

        valid = True

        triu_rows, triu_cols = np.triu_indices(sample.cost.shape[0], k=1)
        cost = sample.cost.values[triu_rows, triu_cols]
        latents = np.hstack((cost, sample.capacity.values, sample.demand.values))

        return torch.Tensor(obs).reshape(1, -1), torch.Tensor(action), discrete_action_placeholder, valid, torch.Tensor(latents).reshape(1, -1)

    def split(self, *split):
        assert len(split) == 3, f'Need 3 splits, got {len(split)}'
        assert np.sum(split) == 1, f'Need splits to sum to 1, got {np.sum(split)}'

        n = self.__len__()
        indices = torch.randperm(n).tolist()
        split_indices = (np.cumsum(split)*n).astype(int)

        train_indices = indices[0:split_indices[0]]
        valid_indices = indices[split_indices[0]:split_indices[1]]
        test_indices =  indices[split_indices[1]:split_indices[2]]

        return torch.utils.data.Subset(self, train_indices), torch.utils.data.Subset(self, valid_indices), torch.utils.data.Subset(self, test_indices)

# if __name__ == '__main__':
#     ds = TaxiDataset('/home/school/Documents/causality-course-project/taxi_dataset_4_1000000.nc')
#     train_set, valid_set, test_set = ds.split(0.7, 0.1, 0.15)
#     train_set[0]