import argparse
import multiprocessing as mp
import numpy as np
import pathlib
import sys
import tqdm
import xarray as xr

from gymnasium.spaces.utils import unflatten as unflatten_space

# sampling à la Learning Causal Representations of Single Cells via Sparse Mechanism Shift Modeling
def sample_intervention(n_nodes, n_interventions, e, rng=None):
    # randomly select 1-n_interventions latents
    n_interventions = 1 if n_interventions == 1 else rng.integers(1, n_interventions)
    idx = rng.choice(2*n_nodes, n_interventions, replace=False)

    # those indices are intervened with a bimodal gaussian distribution
    interventions = np.zeros(2*n_nodes)
    # first, we need to randomly decide if we sample from the positive or negative mean distribution
    es = e*np.sign(rng.uniform(-1, 1, size=n_interventions))
    # then we just sample from it
    interventions[idx] = rng.normal(es, 0.5)
    return interventions

# some hacky shit, but what they do in train.py
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from disentanglement_via_mechanism_sparsity.data import taxi

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n_nodes', type=int, help='Number of samples to generate')
    parser.add_argument('n_samples', type=int, help='Number of samples to generate')
    parser.add_argument('save_path', type=pathlib.Path, help='Where to save dataset')
    parser.add_argument('--perturb_travel_times', action='store_true', help='If set, a capacity change at a node influences travel times to/from that node')
    parser.add_argument('--render_env', action='store_true', help='If set, env will render every 1000 samples')
    parser.add_argument('--samples_per_intervention', type=int, default=1, help='Number of samples to generate with the same intervention')
    parser.add_argument('--dims_per_intervention', type=int, default=3, help='Intervene on 1 to this number of dimensions per intervention')
    parser.add_argument('--intervention_strength', type=float, default=2, help='The mean of the distribution the intervention is sampled from')
    args = parser.parse_args()

    env = taxi.TaxiEnv(args.n_nodes, perturb_travel_times=args.perturb_travel_times)

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
        if i % args.samples_per_intervention == 0:
            intervention = sample_intervention(args.n_nodes, args.dims_per_intervention, args.intervention_strength, rng=env.np_random)
            action = unflatten_space(env.action_space, intervention)

        obs = env.step(action)

        if args.render_env and i % 1000 == 0:
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

    # save generating hyperparams
    metadata = args.__dict__.copy()
    metadata['perturb_travel_times'] = int(metadata['perturb_travel_times'])
    metadata['render_env'] = int(metadata['render_env'])
    del metadata['save_path']
    dataset.attrs = metadata

    dataset.to_netcdf(args.save_path)
