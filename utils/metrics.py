import numpy as np
import torch


class MAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return (y_pred - y_true).abs().mean()

class MSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return ((y_pred - y_true) ** 2).mean()


class MAPE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, eps=1e-9):
        return ((y_pred - y_true).abs() / (eps + y_true.abs())).mean()


class WAPE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, eps=1e-9):
        return (y_pred - y_true).abs().sum() / (eps + y_true.abs()).sum()


class WMAPE(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self._w = weights

    def forward(self, y_pred, y_true, eps=1e-9):
        return ((y_pred - y_true).abs() * self._w).sum() / (eps + y_true.abs()).sum()