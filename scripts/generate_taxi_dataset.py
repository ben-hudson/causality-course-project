import argparse
import numpy as np
import pathlib
import sys
import tqdm
import xarray as xr

# some hacky shit, but what they do in train.py
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from disentanglement_via_mechanism_sparsity.data import taxi

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_nodes', type=int, help='Number of samples to generate')
    parser.add_argument('n_samples', type=int, help='Number of samples to generate')
    parser.add_argument('save_path', type=pathlib.Path, help='Where to save dataset')
    args = parser.parse_args()

    env = taxi.TaxiEnv(args.n_nodes)

    data = {
        'cost': [],
        'capacity': [],
        'demand': [],
        'capacity_perturbation': [],
        'demand_perturbation': [],
        'allocation': [],
        'unused_capacity': [],
        'unserved_demand': [],
    }

    for i in tqdm.trange(args.n_samples):
        action = env.action_space.sample()
        obs = env.step(action)
        if i % 1000 == 0:
            env.render()

        for key, value in obs.items():
            data[key].append(value)

        for key, value in action.items():
            data[key].append(value)

        env.reset()

    dataset = xr.Dataset({
        'cost': xr.DataArray(
            np.array(data['cost']),
            dims=['sample', 'from_node', 'to_node'],
            coords={'sample': np.arange(args.n_samples), 'from_node': np.arange(args.n_nodes), 'to_node': np.arange(args.n_nodes)}
        ),
        'capacity': xr.DataArray(
            np.array(data['capacity']),
            dims=['sample', 'at_node'],
            coords={'sample': np.arange(args.n_samples), 'at_node': np.arange(args.n_nodes)}
        ),
        'demand': xr.DataArray(
            np.array(data['demand']),
            dims=['sample', 'at_node'],
            coords={'sample': np.arange(args.n_samples), 'at_node': np.arange(args.n_nodes)}
        ),
        'capacity_perturbation': xr.DataArray(
            np.array(data['capacity_perturbation']),
            dims=['sample', 'at_node'],
            coords={'sample': np.arange(args.n_samples), 'at_node': np.arange(args.n_nodes)}
        ),
        'demand_perturbation': xr.DataArray(
            np.array(data['demand_perturbation']),
            dims=['sample', 'at_node'],
            coords={'sample': np.arange(args.n_samples), 'at_node': np.arange(args.n_nodes)}
        ),
        'allocation': xr.DataArray(
            np.array(data['allocation']),
            dims=['sample', 'from_node', 'to_node'],
            coords={'sample': np.arange(args.n_samples), 'from_node': np.arange(args.n_nodes), 'to_node': np.arange(args.n_nodes)}
        ),
        'unused_capacity': xr.DataArray(
            np.array(data['unused_capacity']),
            dims=['sample', 'at_node'],
            coords={'sample': np.arange(args.n_samples), 'at_node': np.arange(args.n_nodes)}
        ),
        'unserved_demand': xr.DataArray(
            np.array(data['unserved_demand']),
            dims=['sample', 'at_node'],
            coords={'sample': np.arange(args.n_samples), 'at_node': np.arange(args.n_nodes)}
        ),
    })

    dataset.to_netcdf(args.save_path)
