import argparse
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, DiskSaver, Timer, TerminateOnNan, TimeLimit
from ignite.metrics import RunningAverage
from ignite.contrib.metrics import GpuInfo

from .model import CVAE
from ...metrics import MyMetrics, evaluate_disentanglement, edge_errors
from ...train import get_loader, get_dataset
import torch
import torch.nn as nn

def train(args):
    ## ---- Data ---- ##
    image_shape, cont_c_dim, disc_c_dim, disc_c_n_values, train_dataset, valid_dataset, test_dataset = get_dataset(args)
    train_loader, val_loader, test_loader = get_loader(args, train_dataset, valid_dataset, test_dataset)

    ## ---- Model ---- ##
    model = build_model(args, args.device, image_shape, cont_c_dim, disc_c_dim, disc_c_n_values)
    model = CVAE(image_shape, cont_c_dim, args.z_dim, 64)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

    def loss_fn(something):
        return something

    trainer = create_supervised_trainer(model, optimizer, loss_fn)

    val_metrics = {
        "training_loss": RunningAverage(),
    }
    evaluator = create_supervised_evaluator(model, metrics=val_metrics)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(trainer):
        print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['nll']:.2f}")

    trainer.run(train_loader, max_epochs=100)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    train(args)
