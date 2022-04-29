"""
This module contains utilities for training the models.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.nn import functional as F
from torch.nn import Module
from typing import Callable
import numpy as np


def train(model: Module, optimizer: Optimizer, loss_fn: Callable, train_loader: DataLoader, test_loader: DataLoader,
          epochs: int = 20, device: str = "cpu") -> None:
    """
    Training loop for training a model.

    Args:
        model: model to train.
        optimizer: optimizer to use.
        loss_fn: loss function.
        train_loader: DataLoader for loading the training data.
        test_loader: Dataloader for loading the test data
        epochs: Training epochs. (Default  = 20)
        device: Where to train the model, "cpu" or "cuda". (Default = "cpu")

    Returns:

    """
    for epoch in range(epochs):
        training_loss = 0.0
        test_loss = 0.0
        model.train()

        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
        training_loss /= len(train_loader)

        model.eval()

        num_correct = 0
        num_examples = 0

        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            test_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        test_loss /= len(test_loader)

        print('Epoch: {}, Training Loss: {:.2f}, Test Loss: {:2f}, accuracy = {:.2f}'.format(epoch, training_loss,
                                                                                             test_loss,
                                                                                             num_correct / num_examples))
