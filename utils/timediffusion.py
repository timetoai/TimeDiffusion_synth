import torch
from torch import nn

from .dl import QuantGAN_TemporalBlock

def is_high_freq(time_series, threshold=0.5, rolling_parts=200):
    orig_std = time_series.std().values[0]
    ma_ts = time_series.rolling(len(time_series) // rolling_parts).mean()
    ma_std = ma_ts.std().values[0]
    return abs(ma_std - orig_std) / orig_std > threshold

def ma(time_series, rolling_parts=200, window=None):
    if window is None:
        window = max(len(time_series) // rolling_parts, 2)
    ts1 = time_series.rolling(window, closed="left").mean()
    ts2 = time_series[:: - 1].rolling(window).mean()[:: - 1]
    ts1[ts1.isna()] = ts2[ts1.isna()]
    ts2[ts2.isna()] = ts1[ts2.isna()]
    ats = (ts1 + ts2) / 2
    return ats


class TimeDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn = nn.ModuleList([QuantGAN_TemporalBlock(1, 128, kernel_size=1, stride=1, dilation=1, padding=0, dropout=0.25),
                                 *[QuantGAN_TemporalBlock(128, 128, kernel_size=2, stride=1, dilation=i, padding=i, dropout=0.0)
                                        for i in [2 ** i for i in range(14)]]])
        self.last = nn.Conv1d(128, 1, kernel_size=1, stride=1, dilation=1)

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return x