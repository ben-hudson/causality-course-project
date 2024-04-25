import gymnasium as gym
import io
import itertools
import matplotlib.pyplot as plt
import mip
import networkx as nx
import numpy as np
from matplotlib import patches

from gymnasium import spaces
from scipy.spatial import Delaunay
from scipy.stats.qmc import LatinHypercube

class TaxiEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 0.1}

    def __init__(self, num_nodes, perturb_travel_times=False):
        self.num_nodes = num_nodes
        self.perturb_travel_times = perturb_travel_times

        node_pos = LatinHypercube(2).random(self.num_nodes)
        self.graph = nx.Graph()
        for n, pos in enumerate(node_pos):
            self.graph.add_node(n, pos=pos)

        triangulation = Delaunay(node_pos)
        for tri in triangulation.simplices:
            self.graph.add_edge(tri[0], tri[1])
            self.graph.add_edge(tri[1], tri[2])
            self.graph.add_edge(tri[2], tri[0])

        for i, j in self.graph.edges:
            pos_i = self.graph.nodes[i]['pos']
            pos_j = self.graph.nodes[j]['pos']
            self.graph.edges[i, j]['cost'] = np.linalg.norm(pos_i - pos_j)

        self.default_cost_mean = np.zeros((self.num_nodes, self.num_nodes))
        for i, j in itertools.combinations(self.graph.nodes, 2):
            cost = 10*nx.dijkstra_path_length(self.graph, i, j, weight='cost')
            self.default_cost_mean[i, j] = cost
        self.default_capacity_mean = 5*np.ones(self.num_nodes)
        self.default_demand_mean = 5*np.ones(self.num_nodes)
        self.std = 0.1

        self.observation_space = spaces.Dict({
            'cost': spaces.Box(0, np.inf, shape=(self.num_nodes, self.num_nodes), dtype=np.float32),
            'capacity': spaces.Box(0, np.inf, shape=(self.num_nodes, ), dtype=np.float32),
            'demand': spaces.Box(0, np.inf, shape=(self.num_nodes, ), dtype=np.float32),
            'allocation': spaces.Box(0, np.inf, shape=(self.num_nodes, self.num_nodes), dtype=np.float32),
        })

        self.action_space = spaces.Dict({
            'capacity_perturbation': spaces.Box(-2, 2, shape=(self.num_nodes, ), dtype=np.float32),
            'demand_perturbation': spaces.Box(-2, 2, shape=(self.num_nodes, ), dtype=np.float32)
        })

        self.reset()

    def sample_and_solve_instance(self, cost_mean, capacity_mean, demand_mean):
        # cost_mean is upper triangular, but sampling it also samples zero values
        cost_sample_triu = self.np_random.normal(cost_mean, self.std)
        cost_sample_triu = np.triu(cost_sample_triu)                # reset lower triangular values to 0
        self.cost_sample = cost_sample_triu + cost_sample_triu.T    # make it symmetric
        self.cost_sample = self.cost_sample.clip(0.01)              # min travel cost is 0.01
        np.fill_diagonal(self.cost_sample, 0)                       # ... except on the diagonal, where it is 0

        # capacity and cost cannot be negative
        self.capacity_sample = self.np_random.normal(capacity_mean, self.std).clip(0)
        self.demand_sample = self.np_random.normal(demand_mean, self.std).clip(0)

        model = self._formulate_allocation_problem(self.cost_sample, self.capacity_sample, self.demand_sample)
        assert model.optimize() == mip.OptimizationStatus.OPTIMAL

        self.allocation = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.allocation[i, j] = model.var_by_name(f'y_{i}_{j}').x
        self.unused_capacity = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            self.unused_capacity[i] = model.var_by_name(f'u_{i}').x
        self.unserved_demand = np.zeros(self.num_nodes)
        for i in range(self.num_nodes):
            self.unserved_demand[i] = model.var_by_name(f'v_{i}').x

        return self.allocation, self.unused_capacity, self.unserved_demand

    def _formulate_allocation_problem(self, cost, capacity, demand, unserved_demand_penalty=1e6):
        model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
        model.verbose = 0
        y = model.add_var_tensor((self.num_nodes, self.num_nodes), name='y', var_type='C', lb=0)    # taxis allocated from i->j
        u = model.add_var_tensor((self.num_nodes, ), name='u', var_type='C', lb=0)                  # unused taxis demand at i
        v = model.add_var_tensor((self.num_nodes, ), name='v', var_type='C', lb=0)                  # unserved demand at i

        for n in self.graph.nodes:
            model += mip.xsum(y[n, :]) + u[n] == capacity[n]    # taxis out of node n + unused taxis = capacity
            model += mip.xsum(y[:, n]) + v[n] == demand[n]      # taxis in to node n + unserved demand = demand

        model += mip.xsum((cost * y).flatten()) + mip.xsum(unserved_demand_penalty*v) # elementwise multiplication of cost and y + unserved demand penalties
        return model

    def step(self, action):
        # right now, actions do not have a lasting effect
        capacity_mean = self.capacity_mean + action['capacity_perturbation'] # could change this to a "complex function", like sigmoid
        demand_mean = self.demand_mean + action['demand_perturbation']

        # a positive capacity change causes the travel time to/from a node to increase by 10%
        cost_mean = self.cost_mean.copy()
        if self.perturb_travel_times:
            for i, capacity_change in enumerate(action['capacity_perturbation']):
                cost_mean[i, :] *= (1 + 0.1*capacity_change)
                cost_mean[:, i] *= (1 + 0.1*capacity_change)

        self.sample_and_solve_instance(cost_mean, capacity_mean, demand_mean)

        return {
            'cost': self.cost_sample,
            'capacity': self.capacity_sample,
            'demand': self.demand_sample,
            'allocation': self.allocation,
            'unused_capacity': self.unused_capacity,
            'unserved_demand': self.unserved_demand
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cost_mean = self.default_cost_mean
        self.capacity_mean = self.default_capacity_mean
        self.demand_mean = self.default_demand_mean

    def render(self):
        fig = plt.gcf()
        fig.clear()

        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        self._render(ax)

        if self.render_mode == 'human':
            plt.ion()
            fig.canvas.draw()
            fig.show()

        elif self.render_mode == 'rgb_array':
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            img = data.reshape((int(h), int(w), -1))
            return img

        elif self.render_mode is not None:
            raise Exception(f'render_mode={self.render_mode} is not implemented.')

    def _render(self, ax):
        digraph = self.graph.to_directed()
        node_pos = nx.get_node_attributes(self.graph, 'pos')
        edge_widths = {(i, j): self.allocation[i, j] for i, j in digraph.edges if self.allocation[i, j] > 0}

        # convert capacity/demand to bar height
        scale = max(
            self.default_capacity_mean.max(),
            self.default_demand_mean.max()
        )
        bar_height = 0.1
        bar_width = 0.05
        bar_offset = np.array([0.01, 0])
        for n in digraph.nodes:
            capacity_bar_pos = node_pos[n] + bar_offset
            demand_bar_pos = capacity_bar_pos + np.array([bar_width, 0]) + bar_offset
            ax.add_patch(patches.Rectangle(capacity_bar_pos, bar_width, bar_height*self.capacity_sample[n]/scale, color=plt.cm.tab10(0)))
            ax.add_patch(patches.Rectangle(demand_bar_pos, bar_width, bar_height*self.demand_sample[n]/scale, color=plt.cm.tab10(1)))

        nx.draw_networkx_nodes(digraph, pos=node_pos, node_color='grey')
        nx.draw_networkx_labels(digraph, pos=node_pos, ax=ax)
        nx.draw_networkx_edges(digraph, pos=node_pos, style='dashed', width=0.1, alpha=0.5, arrows=False, ax=ax)
        nx.draw_networkx_edges(digraph, pos=node_pos, edgelist=list(edge_widths.keys()), width=list(edge_widths.values()), ax=ax)

        x_min, _ = ax.get_xlim()
        y_min, _ = ax.get_ylim()
        summary = [
            f'Unused capacity: {self.unused_capacity.sum():.2f}',
            f'Unserved demand: {self.unserved_demand.sum():.2f}',
            f'Reallocation cost: {(self.cost_sample * self.allocation).sum():.2f}'
        ]
        ax.text(x_min + 0.01, y_min + 0.01, '\n'.join(summary))