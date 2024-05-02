
import argparse
import os
import sys
import json
import shutil
import pathlib
import random
from itertools import islice
import time
from copy import deepcopy
import math
from pprint import pprint
import logging

try:
    from comet_ml import Experiment
    COMET_AVAIL = True
except:
    COMET_AVAIL = False

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, Timer, TerminateOnNan, TimeLimit
from ignite.metrics import RunningAverage
from ignite.contrib.metrics import GpuInfo
from ignite.utils import setup_logger

import cooper

# adding the folder containing the folder `disentanglement_via_mechanism_sparsity` to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from disentanglement_via_mechanism_sparsity.universal_logger.logger import UniversalLogger
from disentanglement_via_mechanism_sparsity.metrics import MyMetrics, evaluate_disentanglement, edge_errors
from disentanglement_via_mechanism_sparsity.plot import plot_matrix, plot_weighted_adjacency_vs_steps
from disentanglement_via_mechanism_sparsity.data.synthetic import get_ToyManifoldDatasets
from disentanglement_via_mechanism_sparsity.data.taxi.dataset import TaxiDataset
from disentanglement_via_mechanism_sparsity.model.ilcm_vae import ILCM_VAE
from disentanglement_via_mechanism_sparsity.model.latent_models_vae import FCGaussianLatentModel
from disentanglement_via_mechanism_sparsity.optimization import CustomCMP


def set_manual_seed(opt):
    if opt.seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = opt.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Using seed: {seed}".format(seed=seed))

    if opt.deterministic_cudnn:
        # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
        torch.backends.cudnn.benchmark = False  # may hinder performance
        # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-determinism
        torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)  # makes everything deterministic

def get_dataset(opt):
    if opt.train_prop is None:
        train_prop = 1. - opt.valid_prop - opt.test_prop
    else:
        train_prop = opt.train_prop

    if 'taxi' in opt.dataset:
        print('Using taxi dataset')

        import xarray as xr

        data = xr.open_dataset(pathlib.Path(opt.dataroot) / opt.dataset)
        total_samples = len(data.coords['sample'])
        indices = np.arange(total_samples).tolist() # dataloader shuffles, we don't need to
        split = np.array([train_prop, opt.valid_prop, opt.test_prop])
        split_indices = (np.cumsum(split)*total_samples).astype(int)

        train_indices = indices[0:split_indices[0]]
        val_indices =   indices[split_indices[0]:split_indices[1]]
        test_indices =  indices[split_indices[1]:split_indices[2]]

        train_set = TaxiDataset(data, indices=train_indices, no_norm=opt.no_norm, include_latent_cost=opt.include_latent_cost,
                                include_offsets_in_obs=opt.include_offsets_in_obs)
        val_set = TaxiDataset(data, indices=val_indices, no_norm=opt.no_norm, include_latent_cost=opt.include_latent_cost,
                              include_offsets_in_obs=opt.include_offsets_in_obs)
        test_set = TaxiDataset(data, indices=test_indices, no_norm=opt.no_norm, include_latent_cost=opt.include_latent_cost,
                               include_offsets_in_obs=opt.include_offsets_in_obs)

        obs, cont_c, _, _, _ = train_set[0]
        return (obs.shape[1], ), cont_c.shape[0], 0, [], train_set, val_set, test_set

    assert opt.n_lag <= 1
    manifold, transition_model = opt.dataset.split("-")[-1].split("/")
    datasets = get_ToyManifoldDatasets(manifold, transition_model, split=(train_prop, opt.valid_prop, opt.test_prop),
                                       z_dim=opt.gt_z_dim, x_dim=opt.gt_x_dim, num_samples=opt.num_samples,
                                       no_norm=opt.no_norm, rand_g_density=opt.rand_g_density,
                                       gt_graph_name=opt.gt_graph_name, seed=opt.seed)
    return datasets

def get_loader(opt, train_dataset, valid_dataset, test_dataset):
    train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_workers,
                                   drop_last=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=opt.eval_batch_size, shuffle=False,
                                   num_workers=opt.n_workers,
                                   drop_last=True)
    test_loader = data.DataLoader(test_dataset, batch_size=opt.eval_batch_size, shuffle=False,
                                  num_workers=opt.n_workers,
                                  drop_last=True)
    return train_loader, valid_loader, test_loader


def build_model(opt, device, image_shape, cont_c_dim, disc_c_dim, disc_c_n_values):
    latent_model = FCGaussianLatentModel(opt.z_max_dim, cont_c_dim, disc_c_dim,
                                         disc_c_n_values, opt.network_arch,
                                         n_lag=opt.n_lag,
                                         n_layers=opt.transition_n_layer,
                                         hid_dim=opt.transition_hid_dim,
                                         output_delta=opt.output_delta,
                                         delta_factor=opt.delta_factor,
                                         freeze_m=opt.freeze_m, freeze_g=opt.freeze_g,
                                         freeze_gc=opt.freeze_gc,
                                         freeze_dummies=opt.freeze_dummies,
                                         no_drawhard=opt.no_drawhard,
                                         one_gumbel_sample=opt.one_gumbel_sample,
                                         var_p_mode=opt.var_p_mode, bn=opt.bn_transition_net,
                                         gumbel_temperature=opt.gumbel_temperature,
                                         louizos_gumbel=opt.louizos_gumbel)

    model = ILCM_VAE(latent_model, image_shape, cont_c_dim, disc_c_dim, disc_c_n_values, opt)
    model.to(device)

    return model


def compute_reconstruction_loss(obs, model, reduce=True):
    obs_hat = model.reconstruct(obs)

    if reduce:
        return torch.sum(torch.mean((obs_hat - obs) ** 2, dim=0))
    else:
        return torch.sum((obs_hat - obs) ** 2, dim=[1, 2, 3])


def main(opt):
    # GPU
    device = "cpu" if (not torch.cuda.is_available() or not opt.cuda) else "cuda:0"

    # numerical precision
    if opt.double:
        torch.set_default_tensor_type('torch.DoubleTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    set_manual_seed(opt)

    ## ---- Loggers ---- ##
    if COMET_AVAIL and opt.comet_key is not None and opt.comet_workspace is not None and opt.comet_project_name is not None:
        comet_exp = Experiment(api_key=opt.comet_key, project_name=opt.comet_project_name,
                               workspace=opt.comet_workspace, auto_metric_logging=False, auto_param_logging=False)
        comet_exp.log_parameters(vars(opt))
        if opt.comet_tag is not None:
            comet_exp.add_tag(opt.comet_tag)
    else:
        comet_exp = None
    logger = UniversalLogger(comet=comet_exp,
                             stdout=(not opt.no_print),
                             json=opt.output_dir, throttle=None, max_fig_save=2)

    ## ---- Data ---- ##
    image_shape, cont_c_dim, disc_c_dim, disc_c_n_values, train_dataset, valid_dataset, test_dataset = get_dataset(opt)
    train_loader, valid_loader, test_loader = get_loader(opt, train_dataset, valid_dataset, test_dataset)

    ## ---- Model ---- ##
    model = build_model(opt, device, image_shape, cont_c_dim, disc_c_dim, disc_c_n_values)

    # setting graph to ground-truth
    if opt.set_gc_mask_to_gt:
        if hasattr(train_dataset, "gt_gc"):
            model.latent_model.gc.freeze = True
            model.latent_model.gc.fixed_mask.copy_(train_dataset.gt_gc)
        else:
            print("UserWarning: The attribute `gt_gc` is not provided in the dataset, so --set_gc_mask_to_gt is ignored.")
    if opt.set_g_mask_to_gt:
        if hasattr(train_dataset, "gt_g"):
            model.latent_model.g.freeze = True
            model.latent_model.g.fixed_mask.copy_(train_dataset.gt_g)
        else:
            print("UserWarning: The attribute `gt_g` is not provided in the dataset, so --set_g_mask_to_gt is ignored.")

    if hasattr(train_dataset, "gt_g"):
        max_e_g = train_dataset.gt_g.shape[0] * train_dataset.gt_g.shape[1]
        if opt.set_constraint_to_gt:
            opt.g_constraint = train_dataset.gt_g.sum().item()
    else:
        max_e_g = 0
    if hasattr(train_dataset, "gt_gc"):
        max_e_gc = train_dataset.gt_gc.shape[0] * train_dataset.gt_gc.shape[1]
        if opt.set_constraint_to_gt:
            opt.gc_constraint = train_dataset.gt_gc.sum().item()
    else:
        max_e_gc = 0

    ## ---- Optimization ---- ##
    # setting scaling constants for regularization:
    g_scaling = 1. / int(np.product(image_shape))  # / opt.z_max_dim
    gc_scaling = 1. / int(np.product(image_shape))  # / max(cont_c_dim, 1)

    if opt.lr_dual is None:
        opt.lr_dual = opt.lr

    is_constrained = (opt.g_constraint > 0. or opt.gc_constraint > 0.)
    cmp = CustomCMP(opt.g_reg_coeff, opt.gc_reg_coeff, opt.g_constraint, opt.gc_constraint, g_scaling, gc_scaling,
                    schedule=(opt.constraint_schedule is not None or opt.adaptive_constraint_schedule), max_g=max_e_g, max_gc=max_e_gc)
    formulation = cooper.LagrangianFormulation(cmp)

    if opt.no_adam_gumbel:
        gumbel_params = []
        other_params = []
        for name, param in model.named_parameters():
            if "log_alpha" in name:
                gumbel_params.append(param)
            else:
                other_params.append(param)
        optimizer1 = optim.Adam(other_params, lr=opt.lr, amsgrad=opt.amsgrad)
        optimizer2 = optim.SGD(gumbel_params, lr=opt.lr_gumbel)
        class MultipleOptimizer(object):
            def __init__(self, *op):
                self.optimizers = op

            def zero_grad(self):
                for op in self.optimizers:
                    op.zero_grad()

            def step(self):
                for op in self.optimizers:
                    op.step()
        primal_optimizer = MultipleOptimizer(optimizer1, optimizer2)
    else:
        primal_optimizer = optim.Adam(model.parameters(), lr=opt.lr, amsgrad=opt.amsgrad)

    if is_constrained:
        dual_optimizer = cooper.optim.partial_optimizer(torch.optim.SGD, lr=opt.lr)
    else:
        dual_optimizer = None

    constrained_optimizer = cooper.ConstrainedOptimizer(formulation, primal_optimizer, dual_optimizer,
                                                        dual_restarts=opt.dual_restarts)

    #if opt.scheduler == "reduce_on_plateau":
    #    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
    #                                                            patience=opt.scheduler_patience, threshold=1e-4,
    #                                                            threshold_mode='rel', verbose=True)

    # plot ground-truth graphs
    if hasattr(train_dataset, "gt_gc"):
        fig = plot_matrix(train_dataset.gt_gc, title="Ground-truth G^a", row_label="Z", col_label="A")
        logger.log_figure("gt_ga", fig, step=0)
        plt.close(fig)

    if hasattr(train_dataset, "gt_g"):
        fig = plot_matrix(train_dataset.gt_g, title="Ground-truth G^z", row_label="Z^t", col_label="Z^t-1")
        logger.log_figure("gt_gz", fig, step=0)
        plt.close(fig)


    ## ---- Training Loop ---- ##
    def step(engine, batch):
        if "random" in opt.mode:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        model.train()
        constrained_optimizer.zero_grad()

        # we keep the masks frozen for opt.frozen_masks_period iterations
        if engine.state.iteration <= opt.frozen_masks_period or opt.set_g_mask_to_gt or opt.set_gc_mask_to_gt:
            model.latent_model.freeze_masks()
            if formulation.ineq_multipliers is not None:
                # setting requires_grad to False causes problems with optimizer down the line. Let's just set it to 0.
                torch.nn.init.constant_(formulation.ineq_multipliers.weight, 0.)
        else:
            model.latent_model.unfreeze_masks()

        obs, cont_c, disc_c, valid, other = batch
        obs, cont_c, disc_c, valid, other = obs.to(device), cont_c.to(device), disc_c.to(device), valid.to(device), other.to(device)

        # cooper stuff
        lagrangian = formulation.composite_objective(cmp.closure, model, obs, cont_c, disc_c, valid, other, opt)
        formulation.custom_backward(lagrangian)

        # clip grad
        if opt.max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), opt.max_grad_clip)
        if opt.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)

        # check gradient norm
        grad_norm = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any():
                    print(f"{name} is nan.")
                grad_norm.append(torch.norm(p.grad.data, p=2.))
        grad_norm = torch.norm(torch.stack(grad_norm), p=2.).item()

        # make a step
        constrained_optimizer.step()

        if formulation.ineq_multipliers is not None:
            # when masks are frozen, keep multipliers to zeros.
            if engine.state.iteration <= opt.frozen_masks_period:
                torch.nn.init.constant_(formulation.ineq_multipliers.weight, 0.)
            multiplier = formulation.ineq_multipliers.forward().item()
        else:
            multiplier = 0.

        loss = cmp.state.loss.item()
        defect = 0 if cmp.state.ineq_defect is None else cmp.state.ineq_defect.item()
        g_reg = cmp.state.misc["g_reg"]
        gc_reg = cmp.state.misc["gc_reg"]
        reconstruction_loss = cmp.state.misc["reconstruction_loss"]
        kl = cmp.state.misc["kl"]
        nll = cmp.state.misc["nll"]

        return loss, defect, g_reg, gc_reg, reconstruction_loss, kl, nll, grad_norm, lagrangian.item(), multiplier


    trainer = Engine(step)
    trainer.logger = setup_logger(level=logging.INFO, stream=sys.stdout)

    # keep running average of the training loss
    RunningAverage(output_transform=lambda x: x[0], epoch_bound=False).attach(trainer, "loss_train")
    RunningAverage(output_transform=lambda x: x[1], epoch_bound=False, alpha=1e-6).attach(trainer, "defect")
    RunningAverage(output_transform=lambda x: x[2], epoch_bound=False, alpha=1e-6).attach(trainer, "g_reg")
    RunningAverage(output_transform=lambda x: x[3], epoch_bound=False, alpha=1e-6).attach(trainer, "gc_reg")
    RunningAverage(output_transform=lambda x: x[4], epoch_bound=False, alpha=1e-6).attach(trainer, "reconstruction_loss")
    RunningAverage(output_transform=lambda x: x[5], epoch_bound=False, alpha=1e-6).attach(trainer, "kl")
    RunningAverage(output_transform=lambda x: x[6], epoch_bound=False).attach(trainer, "nll_train")
    RunningAverage(output_transform=lambda x: x[7], epoch_bound=False, alpha=1e-6).attach(trainer, "grad_norm")
    RunningAverage(output_transform=lambda x: x[8], epoch_bound=False, alpha=1e-6).attach(trainer, "lagrangian")
    RunningAverage(output_transform=lambda x: x[9], epoch_bound=False, alpha=1e-6).attach(trainer, "lagrangian_multiplier")

    # makes the code stop if nans are encountered
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    # schedule constraint
    if opt.constraint_schedule is not None:
        @trainer.on(Events.ITERATION_COMPLETED)
        def update_constraint(engine):
            current_g_cst, current_gc_cst = cmp.update_constraint(engine.state.iteration, opt.constraint_schedule, opt.frozen_masks_period)
            logger.log_metrics(step=engine.state.iteration, metrics={"current_g_cst": current_g_cst,
                                                                     "current_gc_cst": current_gc_cst})
    elif opt.adaptive_constraint_schedule is not None:
        @trainer.on(Events.ITERATION_COMPLETED)
        def update_constraint(engine):
            current_g_cst, current_gc_cst = cmp.update_constraint_adaptive(engine.state.iteration,
                                                                           decrease_rate=opt.decrease_rate,
                                                                           no_update_period=opt.frozen_masks_period)
            logger.log_metrics(step=engine.state.iteration, metrics={"current_g_cst": current_g_cst,
                                                                     "current_gc_cst": current_gc_cst})

    # Log GPU info
    if device != "cpu":
        GpuInfo().attach(trainer, name='gpu')  # metric names are 'gpu:X mem(%)', 'gpu:X util(%)'

    ## ---- Evaluation Loop ---- ##
    def eval_step(engine, batch):
        model.eval()

        obs, cont_c, disc_c, valid, other = batch
        obs, cont_c, disc_c, valid, other = obs.to(device), cont_c.to(device), disc_c.to(device), valid.to(device), other.to(device)

        rec_loss, kl, kl_per_dim = 0, 0, torch.zeros((obs.shape[0], opt.z_max_dim)).to(device)
        if opt.mode == "vae":
            log_likelihood, rec_loss, kl, kl_per_dim = model.elbo(obs, cont_c, disc_c)
        elif opt.mode == "supervised_vae":
            obs = obs[:, -1]
            other = other[:, -1]
            z_hat = model.latent_model.mean(model.latent_model.transform_q_params(model.encode(obs)))
            log_likelihood = -torch.mean((z_hat.view(z_hat.shape[0], -1) - other) ** 2, 1)
        elif "random" in opt.mode:
            log_likelihood = torch.zeros((obs.shape[0],)).to(device)
        elif "latent_transition_only" == opt.mode:
            log_likelihood = model.log_likelihood(other, cont_c, disc_c)

        return log_likelihood, valid, rec_loss, kl, kl_per_dim

    # WARNING: These metrics return a dictionary. The behavior is bit weird, we get
    # evaluator.state.metrics = {"all": {"metric1": 0.1, "metric2": 0.2}, "metric1": 0.1, "metric2": 0.2}
    # see this for more: https://github.com/pytorch/ignite/blob/master/ignite/metrics/metric.py#L298
    evaluator = Engine(eval_step)
    evaluator.logger = setup_logger(level=logging.INFO, stream=sys.stdout)
    MyMetrics(opt.include_invalid, "valid").attach(evaluator, "all")

    evaluator_test = Engine(eval_step)
    MyMetrics(opt.include_invalid, "test").attach(evaluator_test, "all")

    ## ---- Checkpointing ---- ##
    to_save = {"model": model} #, "optimizer": optimizer}

    # save initial model, before starting to train
    init_checkpoint_handler = Checkpoint(to_save, DiskSaver(opt.output_dir, create_dir=True), n_saved=1,
                                         filename_prefix="init")
    trainer.add_event_handler(Events.STARTED, init_checkpoint_handler)

    # periodic model saving
    checkpoint_handler = Checkpoint(to_save, DiskSaver(opt.output_dir, create_dir=True), n_saved=1)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=opt.ckpt_period), checkpoint_handler)


    # save model when terminating (because of nan or Time limit)
    terminate_checkpoint_handler = Checkpoint(to_save, DiskSaver(opt.output_dir, create_dir=True), n_saved=1,
                                         filename_prefix="terminate")
    trainer.add_event_handler(Events.TERMINATE, terminate_checkpoint_handler)

    # score we look at for declaring best model
    #def score_function(engine):
    #    """Validation score"""

    #    score = - engine.state.metrics["nll_valid"]

    #    best_checkpoint_handler.filename_prefix = "best"

    #    if opt.best_criterion == "loss":
    #        with torch.no_grad():
    #            g_reg = model.latent_model.g_regularizer().item()
    #            gc_reg = model.latent_model.gc_regularizer().item()
    #        score -= opt.g_reg_coeff * g_reg * g_scaling
    #        score -= opt.gc_reg_coeff * gc_reg * gc_scaling

    #    return score

    # saving if the score improved before thresholding
    #best_checkpoint_handler = Checkpoint(to_save, DiskSaver(opt.output_dir, create_dir=True), n_saved=1,
    #                                     filename_prefix="best",
    #                                     score_function=score_function,
    #                                     score_name=opt.best_criterion + "_valid",
    #                                     greater_or_equal=False)
    #evaluator.add_event_handler(Events.COMPLETED, best_checkpoint_handler)

    ## ---- Timing ---- ##
    timer_avg_iter = Timer(average=True)
    timer_avg_iter.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )
    timer_eval = Timer(average=False)
    timer_eval.attach(
        evaluator,
        start=Events.EPOCH_STARTED,
        resume=Events.EPOCH_STARTED,
        pause=Events.EPOCH_COMPLETED,
        step=Events.EPOCH_COMPLETED,
    )
    timer_total = Timer(average=False)
    timer_total.attach(
        trainer,
        start=Events.STARTED,
        pause=Events.COMPLETED,
    )

    # TimeLimit
    if opt.time_limit is not None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TimeLimit(int(opt.time_limit * 60 * 60)))

    # iter_limit
    if opt.iter_limit is not None:
        @trainer.on(Events.ITERATION_COMPLETED)
        def check_iter_limit(engine):
            if engine.state.iteration >= opt.iter_limit:
                engine.terminate()

    ## ---- Plotting ---- ##
    @trainer.on(Events.ITERATION_COMPLETED(every=opt.plot_period))
    def plot(engine):
        model.eval()
        num_steps = 4
        with torch.no_grad():
            if "vae" in opt.mode:
                if hasattr(train_dataset, "z_names"):
                    z_names = train_dataset.z_names
                else:
                    z_names = []
                perm_mat = None

                # plotting
                if len(evaluator.state.metrics) > 0:
                    mcc, consistent_r, r, cc, C_hat, C_pattern, perm_mat, z, z_hat, transposed_consistent_r = evaluate_disentanglement(model, test_loader, device, opt)
                    fig = plot_matrix(cc, title="Representations correlation matrix", row_label="Ground-truth",
                                         col_label="Learned", row_names=z_names)
                    logger.log_figure("correlation_matrix", fig, step=engine.state.iteration)
                    plt.close(fig)
                    fig = plot_matrix(np.abs(C_hat) / np.max(np.abs(C_hat)), title="Estimated |C|", row_label="Ground-truth",
                                         col_label="Learned", row_names=z_names)
                    logger.log_figure("estimated_C", fig, step=engine.state.iteration)
                    plt.close(fig)
                    fig = plot_matrix(C_pattern, title="Expected C pattern", row_label="Ground-truth",
                                         col_label="Learned", row_names=z_names)
                    logger.log_figure("C_pattern", fig, step=engine.state.iteration)
                    plt.close(fig)

                    # cheap mcc log
                    logger.log_metrics(step=engine.state.iteration, metrics={"mcc": mcc})
                    logger.log_metrics(step=engine.state.iteration, metrics={"consistent_r": consistent_r})
                    logger.log_metrics(step=engine.state.iteration, metrics={"r": r})
                    logger.log_metrics(step=engine.state.iteration, metrics={"transposed_consistent_r": transposed_consistent_r})
                    logger.log_metrics(step=engine.state.iteration, metrics={"nb_nonzero_C": np.count_nonzero(C_pattern)})


                    # plot masks
                    if opt.n_lag > 0:
                        with torch.no_grad():
                            g_prob = model.latent_model.g.get_proba().cpu().numpy()
                        fig = plot_matrix(g_prob, title="G mask", row_label="Z^t", col_label="Z^t-1")
                        logger.log_figure("G_mask", fig, step=engine.state.iteration)
                        plt.close(fig)

                        # same but permute the graph according to optimal MCC permutation
                        if perm_mat is not None:
                            fig = plot_matrix(np.matmul(np.matmul(perm_mat, g_prob), perm_mat.T),
                                                 title="Permuted G mask", row_label="Z^t", col_label="Z^t-1",
                                                 row_names=z_names, col_names=z_names)
                            logger.log_figure("permuted_G_mask", fig, step=engine.state.iteration)
                            plt.close(fig)

                            if not hasattr(engine, "g_probs"):
                                engine.g_probs = []
                                engine.g_probs_iterations = []
                            engine.g_probs.append(g_prob)
                            engine.g_probs_iterations.append(engine.state.iteration)
                            if hasattr(train_dataset, "gt_g"):
                                perm_gt_g = np.matmul(np.matmul(perm_mat.T, train_dataset.gt_g), perm_mat)
                            else:
                                perm_gt_g = np.zeros_like(g_prob)
                            fig = plot_weighted_adjacency_vs_steps(np.stack(engine.g_probs, 0), perm_gt_g, engine.g_probs_iterations)
                            logger.log_figure("G^z", fig, step=engine.state.iteration)
                            plt.close(fig)


                    if cont_c_dim > 0:
                        with torch.no_grad():
                            gc_prob = model.latent_model.gc.get_proba().cpu().numpy()
                        fig = plot_matrix(gc_prob, title="GC mask", row_label="Z", col_label="C")
                        logger.log_figure("GC_mask", fig, step=engine.state.iteration)
                        plt.close(fig)

                        if perm_mat is not None:
                            fig = plot_matrix(np.matmul(perm_mat, gc_prob), title="Permuted GC mask", row_label="Z",
                                                 col_label="C", row_names=z_names)
                            logger.log_figure("permuted_GC_mask", fig, step=engine.state.iteration)
                            plt.close(fig)

                            if not hasattr(engine, "gc_probs"):
                                engine.gc_probs = []
                                engine.gc_probs_iterations = []
                            engine.gc_probs.append(gc_prob)
                            engine.gc_probs_iterations.append(engine.state.iteration)
                            if hasattr(train_dataset, "gt_gc"):
                                perm_gt_gc = np.matmul(perm_mat.T, train_dataset.gt_gc)
                            else:
                                perm_gt_gc = np.zeros_like(gc_prob)
                            fig = plot_weighted_adjacency_vs_steps(np.stack(engine.gc_probs, 0), perm_gt_gc, engine.gc_probs_iterations)
                            logger.log_figure("G^a", fig, step=engine.state.iteration)
                            plt.close(fig)

                    if disc_c_dim > 0:
                        with torch.no_grad():
                            gc_disc_prob = model.latent_model.gc_disc.get_proba().cpu().numpy()
                        fig = plot_matrix(gc_disc_prob, title="GC_disc_mask", row_label="Z", col_label="C")
                        logger.log_figure("GC_disc_mask", fig, step=engine.state.iteration)
                        plt.close(fig)

                        if perm_mat is not None:
                            fig = plot_matrix(np.matmul(perm_mat, gc_disc_prob), title="Permuted GC_disc_mask",
                                                 row_label="Z", col_label="C", row_names=z_names)
                            logger.log_figure("permuted_GC_disc_mask", fig, step=engine.state.iteration)
                            plt.close(fig)

                            if not hasattr(engine, "gc_disc_probs"):
                                engine.gc_disc_probs = []
                                engine.gc_disc_probs_iterations = []
                            engine.gc_disc_probs.append(gc_disc_prob)
                            engine.gc_disc_probs_iterations.append(engine.state.iteration)
                            if hasattr(train_dataset, "gt_gc_disc"):
                                perm_gt_gc_disc = np.matmul(perm_mat.T, train_dataset.gt_gc_disc)
                            else:
                                perm_gt_gc_disc = np.zeros_like(perm_mat)
                            fig = plot_weighted_adjacency_vs_steps(np.stack(engine.gc_disc_probs, 0), perm_gt_gc_disc, engine.gc_disc_probs_iterations)
                            logger.log_figure("G^a (discrete)", fig, step=engine.state.iteration)
                            plt.close(fig)

            if "latent_transition_only" == opt.mode:
                if len(evaluator.state.metrics) > 0:
                    # plot masks
                    if opt.n_lag > 0:
                        with torch.no_grad():
                            g_prob = model.latent_model.g.get_proba().cpu().numpy()
                        fig = plot_matrix(g_prob, title="G mask", row_label="Z^t", col_label="Z^t-1",
                                             row_to_mark=indices)
                        logger.log_figure("G_mask", fig, step=engine.state.iteration)
                        plt.close(fig)
                    if cont_c_dim > 0:
                        with torch.no_grad():
                            gc_prob = model.latent_model.gc.get_proba().cpu().numpy()
                        fig = plot_matrix(gc_prob, title="GC mask", row_label="Z", col_label="C",
                                             row_to_mark=indices)
                        logger.log_figure("GC_mask", fig, step=engine.state.iteration)
                        plt.close(fig)


    ## ---- Logging ---- ##
    @trainer.on(Events.ITERATION_COMPLETED(every=opt.fast_log_period))
    def fast_log(engine):
        metrics = deepcopy(engine.state.metrics)
        logger.log_metrics(step=engine.state.iteration, metrics=metrics)


    @trainer.on(Events.ITERATION_COMPLETED(every=opt.eval_period))
    def evaluate_and_log(engine):
        with torch.no_grad():
            evaluator.run(valid_loader, 1)

        # make sure no keys intersect before concatenating
        assert len(set(engine.state.metrics.keys()) & set(evaluator.state.metrics["all"].keys())) == 0
        metrics = deepcopy(engine.state.metrics)
        metrics.update(evaluator.state.metrics["all"])

        # best metrics
        #g_reg = opt.g_reg_coeff * g_scaling * model.latent_model.g_regularizer().item()
        #gc_reg = opt.gc_reg_coeff * gc_scaling * model.latent_model.gc_regularizer().item()
        #total_reg =  g_reg + gc_reg

        #best_score = best_checkpoint_handler._saved[-1].priority

        #if opt.best_criterion == "loss":
        #    metrics["best_loss_valid"] = - best_score
        #    metrics["best_nll_valid"] = - best_score - total_reg
        #if opt.best_criterion == "nll":
        #    metrics["best_loss_valid"] = - best_score + total_reg
        #    metrics["best_nll_valid"] = - best_score

        # LR scheduling
        #if opt.scheduler == "reduce_on_plateau":
        #    scheduler.step(- best_score)
        #metrics["last_lr"] = scheduler._last_lr[0]

        # timers
        metrics["time_avg_iter"] = timer_avg_iter.value()
        timer_avg_iter.reset()
        metrics["time_eval"] = timer_eval.value()
        metrics["time_total"] = timer_total.value()

        logger.log_metrics(step=engine.state.iteration, metrics=metrics)


    @trainer.on(Events.COMPLETED)
    def final_evaluate_and_log(engine):
        """
        Post-fix == _best : Best model before thresholding
        post-fix == _final: Best model after thresholding and further training.
        """
        print("final_evaluate_and_log...")
        with torch.no_grad():
            evaluator.run(valid_loader, 1)
            evaluator_test.run(test_loader, 1)

        # merge metrics of trainer, evaluator and evaluator_test
        assert len(set(engine.state.metrics.keys()) & set(evaluator.state.metrics["all"].keys()) & set(evaluator_test.state.metrics["all"].keys())) == 0
        metrics = deepcopy(engine.state.metrics)
        metrics.update(evaluator.state.metrics["all"])
        metrics.update(evaluator_test.state.metrics["all"])

        # add _final postfix
        for key in list(metrics.keys()):
            metrics[key + "_final"] = metrics.pop(key)

        # best metrics
        #g_reg = opt.g_reg_coeff * g_scaling * model.latent_model.g_regularizer().item()
        #gc_reg = opt.gc_reg_coeff * gc_scaling * model.latent_model.gc_regularizer().item()
        #total_reg = g_reg + gc_reg

        #best_score = best_checkpoint_handler._saved[-1].priority

        #if opt.best_criterion == "loss":
        #    metrics["best_loss_valid"] = - best_score
        #    metrics["best_nll_valid"] = - best_score - total_reg
        #if opt.best_criterion == "nll":
        #    metrics["best_loss_valid"] = - best_score + total_reg
        #    metrics["best_nll_valid"] = - best_score

        # timers
        metrics["time_avg_iter"] = timer_avg_iter.value()
        timer_avg_iter.reset()
        metrics["time_eval"] = timer_eval.value()
        metrics["time_total"] = timer_total.value()

        # misc
        metrics["num_examples_train"] = len(train_loader.dataset)

        if opt.mode != "latent_transition_only" :
            mcc, consistent_r, r, cc, C_hat, C_pattern, perm_mat, z, z_hat, transposed_consistent_r = evaluate_disentanglement(model, test_loader, device, opt)
            metrics["mean_corr_coef_final"] = mcc
            metrics["consistent_r_final"] = consistent_r
            metrics["r_final"] = r
            metrics["transposed_consistent_r_final"] = transposed_consistent_r

            # Evaluate linear_score and MCC on best models after thresholding
            #best_files = [f.name for f in os.scandir(opt.output_dir) if f.name.startswith("best")]
            #if len(best_files) > 0:
            #    print(f"Found {len(best_files)} best checkpoints, evaluating the last one.")
            #    model.load_state_dict(torch.load(os.path.join(opt.output_dir, best_files[-1])))  #["model"])
            #    model.eval()
            #else:
            #    print(f"Found 0 thresh_best checkpoints, reporting final metric")
            #mcc, consistent_r, r, cc, C_hat, C_pattern, perm_mat, z, z_hat = evaluate_disentanglement(model, test_loader, device, opt)
            #metrics["mean_corr_coef_best"] = mcc
            #metrics["consistent_r_best"] = consistent_r
            #metrics["r_best"] = r

            # save both ground_truth and learned latents
            np.save(os.path.join(opt.output_dir, "z_hat_final.npy"), z_hat)
            np.save(os.path.join(opt.output_dir, "z_gt_final.npy"), z)
        else:
            perm_mat = np.eye(opt.z_max_dim)

        if hasattr(train_dataset, "gt_g") and hasattr(model.latent_model, "g"):
            learned_g = (model.latent_model.g.get_proba() > 0.5).cpu().numpy().astype(np.float32)
            permuted_learned_g = np.matmul(np.matmul(perm_mat, learned_g), perm_mat.transpose())
            #metrics["g_fn"], metrics["g_fp"] = edge_errors(permuted_learned_g,  train_dataset.gt_g.cpu().numpy())
            # Some runs used the above where target and prediction are flipped, resulting in flipped fn and fp
            metrics["g_fn"], metrics["g_fp"] = edge_errors(train_dataset.gt_g.cpu().numpy(), permuted_learned_g)
            metrics["g_shd"] = metrics["g_fn"] + metrics["g_fp"]
        if hasattr(train_dataset, "gt_gc") and hasattr(model.latent_model, "gc"):
            learned_gc = (model.latent_model.gc.get_proba() > 0.5).cpu().numpy().astype(np.float32)
            permuted_learned_gc = np.matmul(perm_mat, learned_gc)
            #metrics["gc_fn"], metrics["gc_fp"] = edge_errors(permuted_learned_gc,  train_dataset.gt_gc.cpu().numpy())
            # Some runs used the above where target and prediction are flipped, resulting in flipped fn and fp
            metrics["gc_fn"], metrics["gc_fp"] = edge_errors(train_dataset.gt_gc.cpu().numpy(), permuted_learned_gc)
            metrics["gc_shd"] = metrics["gc_fn"] + metrics["gc_fp"]

        logger.log_metrics(step=engine.state.iteration, metrics=metrics)

    # start training
    trainer.run(train_loader, opt.epochs)


def init_exp(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True, choices=["vae", "supervised_vae", "random_vae", "latent_transition_only"],
                        help="Model to use")
    parser.add_argument("--full_seq", action="store_true",
                        help="This flag can only be used with --mode vae and --n_lag == 1. \
                              It makes the model expect the examples in the minibatch to be full sequences.")
    # data
    parser.add_argument("--dataset", type=str, required=True,
                        help="Type of the dataset to be used. 'toy-MANIFOLD/TRANSITION_MODEL'")
    parser.add_argument("--gt_z_dim", type=int, default=10,
                        help="ground truth dimensionality of z (for TRANSITION_MODEL == 'temporal_sparsity_non_trivial')")
    parser.add_argument("--gt_x_dim", type=int, default=20,
                        help="ground truth dimensionality of x (for MANIFOLD == 'nn')")
    parser.add_argument("--rand_g_density", type=float, default=None,
                        help="Probability of sampling an edge. When None, the graph is set to a default (or to gt_graph_name).")
    parser.add_argument("--gt_graph_name", type=str, default=None,
                        help="Name of the ground-truth graph to use in synthetic data.")
    parser.add_argument("--num_samples", type=int, default=int(1e6),
                        help="num_samples for synthetic datasets")
    parser.add_argument("--no_norm", action="store_true",
                        help="no normalization in toy datasets")
    parser.add_argument("--dataroot", type=str, default="./",
                        help="path to dataset")
    parser.add_argument("--include_latent_cost", action="store_true",
                        help="taxi dataset: include cost parameters in latents")
    parser.add_argument("--include_offsets_in_obs", action="store_true",
                        help="taxi dataset: include unused capacity and unserved demand in observation")
    parser.add_argument("--train_prop", type=float, default=None,
                        help="proportion of all samples used in validation set")
    parser.add_argument("--valid_prop", type=float, default=0.10,
                        help="proportion of all samples used in validation set")
    parser.add_argument("--test_prop", type=float, default=0.10,
                        help="proportion of all samples used in test set")
    parser.add_argument("--include_invalid", action="store_true",
                        help="proportion of all samples used in test set")
    parser.add_argument("--n_workers", type=int, default=2,
                        help="number of data loading workers")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="batch size used during training")
    parser.add_argument("--eval_batch_size", type=int, default=1024,
                        help="batch size used during evaluation")
    parser.add_argument("--epochs", type=int, default=500,
                        help="number of epochs to train for")
    parser.add_argument("--time_limit", type=float, default=None,
                        help="After this amount of time, terminate training.")
    parser.add_argument("--iter_limit", type=int, default=None,
                        help="After this amount of iteration, terminate training.")

    # identifiable latent causal model (ILCM)
    parser.add_argument("--network_arch", type=str, default="MLP",
                        choices=["MLP"],
                        help="Type of network used for the transition model.")
    parser.add_argument("--n_lag", type=int, default=1,
                        help="p(x_t | x_t-1, ..., x_t-n_lag)")
    parser.add_argument("--z_dim", type=int, default=10,
                        help="Dimension of the learned latent representation")
    parser.add_argument("--transition_n_layer", type=int, default=5,
                        help="number of hidden layers in transition NNs (0 implies linear).")
    parser.add_argument("--transition_hid_dim", type=int, default=512,
                        help="number of units in each hidden layers of the transition NNs.")
    parser.add_argument("--output_delta", action="store_true",
                        help="In transition model, the net will output the delta between z_tm1 and z_t instead of z_t")
    parser.add_argument("--delta_factor", type=float, default=1.,
                        help="factor multiplying the delta outputted by the transition network. (useful only with --output_delta)")
    parser.add_argument("--g_reg_coeff", type=float, default=0.0,
                        help="Regularization coefficient for graph connectivity between z^t and z^{<t}")
    parser.add_argument("--gc_reg_coeff", type=float, default=0.0,
                        help="Regularization coeff for graph connectivity between z^t and c")
    parser.add_argument("--g_constraint", type=float, default=0.0,
                        help="Constrain G^z to have no more than this number of edges")
    parser.add_argument("--gc_constraint", type=float, default=0.0,
                        help="Constrain G^a to have no more than this number of edges")
    parser.add_argument("--set_constraint_to_gt", action="store_true",
                        help="Will set the maximal number of edges to the number of edges in the ground-truth.")
    parser.add_argument("--drawhard", action="store_true",
                        help="Instead of using soft samples in gumbel sigmoid, use hard samples in forward.")
    #parser.add_argument("--one_gumbel_sample", action="store_true",
    #                    help="Use only one sample of the gumbel masks per minibatch.")
    parser.add_argument("--many_gumbel_sample", action="store_true",
                        help="Use multiple samples of the gumbel masks per minibatch.")
    parser.add_argument("--gumbel_temperature", type=float, default=1.0,
                        help="Controls the temperature in the gumbel-sigmoid masks.")
    parser.add_argument("--louizos_gumbel", action="store_true",
                        help="Use thresholded gumbel sigmoid from Louizos et al. (2018)")
    parser.add_argument("--freeze_m", action="store_true",
                        help="Do not learn m")
    parser.add_argument("--freeze_g", action="store_true",
                        help="Do not learn g")
    parser.add_argument("--freeze_gc", action="store_true",
                        help="Do not learn gc")
    parser.add_argument("--set_gc_mask_to_gt", action="store_true",
                        help="Do not learn gc mask, fix it to ground truth value.")
    parser.add_argument("--set_g_mask_to_gt", action="store_true",
                        help="Do not learn g mask, fix it to ground truth value.")
    parser.add_argument("--unfreeze_dummies", action="store_true",
                        help="Learn the dummy parameters in masking")
    parser.add_argument("--var_p_mode", type=str, default="independent", choices=["dependent", "independent", "fixed"],
                        help="dependent: dependency on z^t-1, independent: no dep on z^t-1, fixed: not learned at all")
    #parser.add_argument("--learn_decoder_var", action="store_true",
    #                    help="learn a variance of p(x|z)")
    parser.add_argument("--fix_decoder_var", action="store_true",
                        help="do not learn the variance of p(x|z)")
    parser.add_argument("--init_decoder_var", type=float, default=None,
                        help="The initial variance of p(x|z).")
    parser.add_argument("--bn_enc_dec", action="store_true",
                        help="Whether to use batch norm or not in encoder/decoder.")
    parser.add_argument("--bn_transition_net", action="store_true",
                        help="Whether to use batch norm or not is transition net.")

    # vae
    parser.add_argument("--encoder", type=str, default='tabular', choices=['tabular'],
                        help="VAE encoder architecture")
    parser.add_argument("--decoder", type=str, default='tabular', choices=['tabular'],
                        help="VAE decoder architecture")
    parser.add_argument("--encoder_depth_multiplier", type=int, default=2,
                        help="The amount of channels per layer is multiplied by this value")
    parser.add_argument("--decoder_depth_multiplier", type=int, default=2,
                        help="The amount of channels per layer is multiplied by this value")
    parser.add_argument("--beta", type=float, default=1)

    # optimization
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--no_adam_gumbel", action="store_true",
                        help="Use SGD for gumbel parameters and Adam for the rest.")
    parser.add_argument("--lr_gumbel", type=float, default=1.0, help="Learning rate")
    parser.add_argument("--lr_dual", type=float, default=None, help="Learning rate")
    parser.add_argument("--dual_restarts", action="store_true",
                        help="When doing constrained optim, restarts the dual value to zero as soon as the constraint is satisfied")
    parser.add_argument("--frozen_masks_period", type=int, default=0,
                        help="Number of iterations we keep masks frozen")
    parser.add_argument("--constraint_schedule", type=int, default=None,
                        help="When specified, the upper bound on number of edges will linearly decrease taking this number of iterations.")
    parser.add_argument("--adaptive_constraint_schedule", action="store_true",
                        help="The upper bound is progressively reduced. decrease progressively, only when defects < 0.1")
    parser.add_argument("--decrease_rate", type=float, default=0.0005,
                        help="Decreasing rate of the constraint when --adaptive_constraint_schedule.")
    parser.add_argument("--max_grad_clip", type=float, default=0,
                        help="Max gradient value (clip above - for off)")
    parser.add_argument("--max_grad_norm", type=float, default=0,
                        help="Max norm of gradient (clip above - 0 for off)")
    parser.add_argument("--amsgrad", action="store_true",
                        help="Use AMSgrad instead of Adam.")


    # logging
    parser.add_argument("--output_dir", required=True,
                        help="Directory to output logs and model checkpoints")
    parser.add_argument("--fresh", action="store_true",
                        help="Remove output directory before starting, even if experiment is completed.")
    parser.add_argument("--ckpt_period", type=int, default=50000,
                        help="Number of batch iterations between each checkpoint.")
    parser.add_argument("--eval_period", type=int, default=5000,
                        help="Number of batch iterations between each evaluation on the validation set.")
    parser.add_argument("--fast_log_period", type=int, default=100,
                        help="Number of batch iterations between each cheap log.")
    parser.add_argument("--plot_period", type=int, default=10000,
                        help="Number of batch iterations between each cheap log.")
    parser.add_argument("--scheduler", type=str, default="reduce_on_plateau", choices=["reduce_on_plateau"],
                        help="Patience for reducing the learning rate in terms of evaluations on tye validation set")
    parser.add_argument("--scheduler_patience", type=int, default=120,
                        help="(applies only to reduce_on_plateau) Patience for reducing the learning rate in terms of evaluations on tye validation set")
    parser.add_argument("--best_criterion", type=str, default="loss", choices=["loss", "nll"],
                        help="Criterion to look at for saving best model and early stopping. loss include regularization terms")
    parser.add_argument('--no_print', action="store_true",
                        help='do not print')
    parser.add_argument('--comet_key', type=str, default=None,
                        help="comet api-key")
    parser.add_argument('--comet_tag', type=str, default=None,
                        help="comet tag, to ease comparison")
    parser.add_argument('--comet_workspace', type=str, default=None,
                        help="comet workspace")
    parser.add_argument('--comet_project_name', type=str, default=None,
                        help="comet project_name")


    # misc
    parser.add_argument("--no_cuda", action="store_false", dest="cuda",
                        help="Disables cuda")
    parser.add_argument("--double", action="store_true",
                        help="Use Double precision")
    parser.add_argument("--seed", type=int, default=None,
                        help="manual seed")
    parser.add_argument("--deterministic_cudnn", action="store_true",
                        help="Forces determinism in cudnn, for greater reproducibilty on GPU.")


    if args is not None:
        opt = parser.parse_args(args)
    else:
        opt = parser.parse_args()

    # option preparation
    opt.num_samples = int(opt.num_samples)  # cast float to integer
    opt.no_drawhard = not opt.drawhard
    opt.freeze_dummies = not opt.unfreeze_dummies
    opt.z_max_dim = opt.z_dim
    opt.freeze_m = True
    opt.learn_decoder_var = not opt.fix_decoder_var
    opt.one_gumbel_sample = not opt.many_gumbel_sample

    # Hack to get a plot of the random representation and avoid training altogether
    if "random" in opt.mode:
        opt.plot_period = 1
        opt.time_limit = 0.001

    # create experiment path
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    # verify if terminate_ file exists, meaning the experiment is already completed. In that case, end script.
    if not opt.fresh:
        with os.scandir(opt.output_dir) as it:
            for entry in it:
                if entry.name.startswith("terminate_model") and entry.name.endswith(".pt"):
                    print("This experiment is already completed and --fresh is False. Ending program.")
                    sys.exit()  # stop program

    # wiping out experiment folder
    for filename in os.listdir(opt.output_dir):
        file_path = os.path.join(opt.output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    kwargs = vars(opt)
    del kwargs["fresh"]

    with open(os.path.join(opt.output_dir, "hparams.json"), "w") as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    return opt

if __name__ == "__main__":
    opt = init_exp()
    main(opt)
