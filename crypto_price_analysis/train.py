import time
import typing
import collections.abc

import torch
from torch.utils.data import DataLoader


class StatusLogger:

    msg_len = 0

    @classmethod
    def update(cls, msg):
        padding = cls.msg_len - len(msg) if len(msg) < cls.msg_len else 0
        print('\r' + msg + ' ' * padding, end='')
        cls.msg_len = len(msg)

    @classmethod
    def print(cls, *args, **kwargs):
        msg = ' '.join(map(str, args))
        print(msg, **kwargs)
        cls.msg_len = len(msg)


def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_function):

    model.train()

    running_loss = 0
    total_samples = 0

    for b, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        StatusLogger.update(f'batch_progress: {b / len(loader):.2%} loss: {running_loss / total_samples:.5f}')

    return running_loss / total_samples


def evaluate_model(model: torch.nn.Module, loader: DataLoader, loss_function):

    model.eval()

    running_loss = 0

    with torch.no_grad():
        for x, y in loader:
            y_pred = model(x)
            loss = loss_function(y_pred, y).item()
            running_loss += loss * x.size(0)

    return running_loss / len(typing.cast(collections.abc.Sized, loader.sampler))


def train_model(model: torch.nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_function,
                epochs: int,
                scheduler = None,
                val_loss_function = None):

    train_history = []
    val_history = []

    val_loss_function = val_loss_function or loss_function

    for e in range(epochs):
        start_time = time.time()

        avg_train_loss = train_epoch(model, train_loader, optimizer, loss_function)
        train_history.append(avg_train_loss)

        if scheduler:
            scheduler.step()

        avg_val_loss = evaluate_model(model, val_loader, val_loss_function)
        val_history.append(avg_val_loss)

        StatusLogger.update(f'epoch: {e:4d} duration: {time.time() - start_time:.2f}s loss: {avg_train_loss:.5f} val_loss: {avg_val_loss:.5f}')
        StatusLogger.print()

    return train_history, val_history
