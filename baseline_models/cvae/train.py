import argparse
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn

from comet_ml import Experiment
from ignite.contrib.metrics import GpuInfo
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, DiskSaver, Timer, TerminateOnNan, TimeLimit
from ignite.metrics import RunningAverage

from ...metrics import MyMetrics, evaluate_disentanglement, edge_errors
from ...train import get_loader, get_dataset
from ...universal_logger.logger import UniversalLogger
from .model import CVAE

StepOutput = namedtuple("StepOutput", "obs obs_hat mse kld loss")

def train(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.comet_key is not None and args.comet_workspace is not None and args.comet_project_name is not None:
        comet_exp = Experiment(api_key=args.comet_key, project_name=args.comet_project_name,
                               workspace=args.comet_workspace, auto_metric_logging=False, auto_param_logging=False)
        comet_exp.log_parameters(vars(args))
        if args.comet_tag is not None:
            comet_exp.add_tag(args.comet_tag)
    else:
        comet_exp = None
    logger = UniversalLogger(comet=comet_exp,
                             stdout=(not args.no_print),
                             json=args.output_dir, throttle=None, max_fig_save=2)

    ## ---- Data ---- ##
    image_shape, cont_c_dim, disc_c_dim, disc_c_n_values, train_dataset, valid_dataset, test_dataset = get_dataset(args)
    train_loader, val_loader, test_loader = get_loader(args, train_dataset, valid_dataset, test_dataset)

    model = CVAE(image_shape, cont_c_dim, args.z_dim, args.hidden_dim)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

    def loss_fn(something):
        return something

    def step(trainer, batch):
        model.train()
        optimizer.zero_grad()

        obs, cond, _, _, _ = batch
        obs, cond = obs.to(device), cond.to(device)

        prior, posterior, obs_hat, mse, kld, loss = model(obs, cond)

        loss.backward()
        optimizer.step()

        return StepOutput(obs.cpu(), obs_hat.cpu(), mse.cpu().item(), kld.cpu().item(), loss.cpu().item())

    trainer = Engine(step)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training_loss(trainer):
        print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output.loss:.2f}")

    def evaluate_and_log_metrics(model, loader, name, it, logger, device, args):
        model.eval()
        mcc, consistent_r, r, cc, C_hat, C_pattern, perm_mat, z, z_hat, transposed_consistent_r = evaluate_disentanglement(model, loader, device, args)
        logger.log_metrics(step=it, metrics={name + "_mcc": mcc})
        logger.log_metrics(step=it, metrics={name + "_consistent_r": consistent_r})
        logger.log_metrics(step=it, metrics={name + "_r": r})
        logger.log_metrics(step=it, metrics={name + "_transposed_consistent_r": transposed_consistent_r})
        logger.log_metrics(step=it, metrics={name + "_nb_nonzero_C": np.count_nonzero(C_pattern)})

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluate_and_log_metrics(model, train_loader, 'train', trainer.state.iteration, logger, device, args)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluate_and_log_metrics(model, val_loader, 'val', trainer.state.iteration, logger, device, args)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluate_and_log_metrics(model, test_loader, 'test', trainer.state.iteration, logger, device, args)

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_validation_results(trainer):
    #     evaluator.run(val_loader)
    #     metrics = evaluator.state.metrics
    #     print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    trainer.run(train_loader, max_epochs=args.epochs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=64, help='Max hidden dim for CVAE')
    parser.add_argument('--mode', choices=['vae'], default='vae', help='For compatibility')

    parser.add_argument("--dataset", type=str, required=True,
                        help="Type of the dataset to be used. 'toy-MANIFOLD/TRANSITION_MODEL'")
    parser.add_argument("--no_norm", action="store_true",
                        help="no normalization in toy datasets")
    parser.add_argument("--dataroot", type=str, default="./",
                        help="path to dataset")
    parser.add_argument("--include_latent_cost", action="store_true",
                        help="taxi dataset: include cost parameters in latents")
    parser.add_argument("--include_offsets_in_obs", action="store_true",
                        help="taxi dataset: include unused capacity and unserved demand in observation")
    parser.add_argument("--valid_prop", type=float, default=0.10,
                        help="proportion of all samples used in validation set")
    parser.add_argument("--test_prop", type=float, default=0.10,
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

    args = parser.parse_args()
    train(args)
