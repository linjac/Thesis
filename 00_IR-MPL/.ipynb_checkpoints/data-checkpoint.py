from io import open
import unicodedata
import re
import pickle
import os
import os.path

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import download_and_extract_archive

import math
import timeit
import numpy as np

# import librosa
# import resampy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from IPython.display import Audio, display

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
    
DEFAULT_OFFSET = 201

import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

class MeshRIRDataset_S32_M1(Dataset):
    x_file = "S32-M1_posSrc.npy"
    y_file = "S32-M1_ir.npy"
    sample_rate = 48000
    DEFAULT_OFFSET = 201
    ir_len = 0
    
    def __init__(self, root, train=None, resample=None):
        self.root = root
        self._folder = folder = os.path.join(root, 'MeshRIR')
        print(folder)
        print(open(os.path.join(folder, self.x_file)))
        self.x = torch.from_numpy(np.load(open(os.path.join(folder, self.x_file), "rb")))
        self.y = torch.from_numpy(np.load(open(os.path.join(folder, self.y_file), "rb")))
        if resample!=None:
            self.y = resample_waveform(self.y, self.sample_rate, resample)
            self.sample_rate = resample
        self.ir_len = self.y.shape[-1]
        self.pairs = None
        self._preprocess(train)

    def _preprocess(self, train):
        pairs = self._make_pairs()
        train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=1, shuffle=True)
        self.pairs = train_pairs if train else test_pairs           
    def __len__(self):
        return len(self.pairs)   
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
#         plot_sweep(pair[1,None],self.sample_rate,"Example IR", max_sweep_rate=48000, offset=self.DEFAULT_OFFSET)
        return (pair[0], pair[1])

    def _make_pairs(self):
        return list(zip(self.x, self.y))
    
    def get_ir_len(self):
        print(self.ir_len)
        return self.ir_len



def _get_log_freq(sample_rate, max_sweep_rate, offset):
    """Get freqs evenly spaced out in log-scale, between [0, max_sweep_rate // 2]

    offset is used to avoid negative infinity `log(offset + x)`.

    """
    start, stop = math.log(offset), math.log(offset + max_sweep_rate // 2)
    return torch.exp(torch.linspace(start, stop, sample_rate, dtype=torch.double)) - offset


def _get_inverse_log_freq(freq, sample_rate, offset):
    """Find the time where the given frequency is given by _get_log_freq"""
    half = sample_rate // 2
    return sample_rate * (math.log(1 + freq / offset) / math.log(1 + half / offset))


def _get_freq_ticks(sample_rate, offset, f_max):
    # Given the original sample rate used for generating the sweep,
    # find the x-axis value where the log-scale major frequency values fall in
    times, freq = [], []
    for exp in range(2, 5):
        for v in range(1, 10):
            f = v * 10**exp
            if f < sample_rate // 2:
                t = _get_inverse_log_freq(f, sample_rate, offset) / sample_rate
                times.append(t)
                freq.append(f)
    t_max = _get_inverse_log_freq(f_max, sample_rate, offset) / sample_rate
    times.append(t_max)
    freq.append(f_max)
    return times, freq


def get_sine_sweep(sample_rate, offset=DEFAULT_OFFSET):
    max_sweep_rate = sample_rate
    freq = _get_log_freq(sample_rate, max_sweep_rate, offset)
    delta = 2 * math.pi * freq / sample_rate
    cummulative = torch.cumsum(delta, dim=0)
    signal = torch.sin(cummulative).unsqueeze(dim=0)
    return signal


def plot_sweep(
    waveform,
    sample_rate,
    title,
    max_sweep_rate=48000,
    offset=DEFAULT_OFFSET,
):
    x_ticks = [100, 500, 1000, 5000, 10000, 20000, max_sweep_rate // 2]
    y_ticks = [1000, 5000, 10000, 20000, sample_rate // 2]

    time, freq = _get_freq_ticks(max_sweep_rate, offset, sample_rate // 2)
    freq_x = [f if f in x_ticks and f <= max_sweep_rate // 2 else None for f in freq]
    freq_y = [f for f in freq if f in y_ticks and 1000 <= f <= sample_rate // 2]

    figure, axis = plt.subplots(1, 1)
    _, _, _, cax = axis.specgram(waveform[0].numpy(), Fs=sample_rate)
    plt.xticks(time, freq_x)
    plt.yticks(freq_y, freq_y)
    axis.set_xlabel("Original Signal Frequency (Hz, log scale)")
    axis.set_ylabel("Waveform Frequency (Hz)")
    axis.xaxis.grid(True, alpha=0.67)
    axis.yaxis.grid(True, alpha=0.67)
    figure.suptitle(f"{title} (sample rate: {sample_rate} Hz)")
    plt.colorbar(cax)
    plt.show(block=True)
    
def resample_waveform(waveform, sample_rate, resample_rate):
    resampled_waveform = F.resample(
        waveform,
        sample_rate,
        resample_rate,
        lowpass_filter_width=16,
        rolloff=0.85,
        resampling_method="sinc_interp_kaiser",
        beta=8.555504641634386,
    )
    print("Resampling waveform from ", sample_rate, " to ", resample_rate)
    print(waveform[0, None].shape)
    print(resampled_waveform[0, None].shape)
    plot_sweep(waveform[0, None], sample_rate, title="Original Waveform")
    plot_sweep(resampled_waveform[0, None], resample_rate, title="Resampled Waveform")
    return resampled_waveform
#     Audio(resampled_waveform.numpy()[0], rate=resample_rate)

# import tools
# data_dir = tools.select_data_dir()
# trainset = MeshRIRDataset_S32_M1(root=data_dir, train=True, resample=16000)