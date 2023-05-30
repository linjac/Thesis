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
from scipy.io import wavfile

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
    
DEFAULT_OFFSET = 201
byte_scale = 2147483647
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

class MotusDataset_best_of(Dataset):
    sample_rate = 48000
    ir_len = 0
    
    def __init__(self, root, train=None, cutoff=0.05, num_tail_samples=None, spectrogram=False, resample=None):
        """
        INPUTS
            train: bool - generate train set or test set
            cutoff: length of head in seconds
            num_tail_samples: length of tail in samples
            spectrogram: bool - generate spectrogram representation
            resample: resample audio
        """
        
        self.root = root
        self._folder = folder = os.path.join(root, 'Motus/best_of/sh_rirs')
        print(folder)
        self.spectrogram=spectrogram
        self.cutoff = cutoff

#         # cutoff: length of head in seconds
#         cutoff_idx = round(cutoff*self.sample_rate)
        
        # num_tail_samples: length of tail in samples
        if num_tail_samples!=None:
            tail_idx_end = cutoff_idx + num_tail_samples
        else:
            tail_idx_end = None
        
        # read waveform tensors
#         temp_head = []
#         temp_tail = []
#         for filename in os.listdir(folder):
#             fs, wav = wavfile.read(os.path.join(folder, filename))
#             temp_head.append(torch.from_numpy(wav[:cutoff_idx,0].T/byte_scale))
#             temp_tail.append(torch.from_numpy(wav[cutoff_idx:tail_idx_end,0].T/byte_scale))
#         # stack extracted waveform tensors
#         self.x = torch.stack(temp_head,dim=0).float()
#         self.y = torch.stack(temp_tail,dim=0).float()

        temp = []
        for filename in os.listdir(folder):
            fs, wav = wavfile.read(os.path.join(folder, filename))
            temp.append(torch.from_numpy(wav[:tail_idx_end,0].T/byte_scale))
        self.data = torch.stack(temp,dim=0).float()
        self.sample_rate = fs
        
        # scale entire dataset to between 1 and -1
        scale_factor = torch.max(torch.abs(self.data))
        self.data = self.data/scale_factor
        
        if resample!=None:
            self.data = resample_waveform(self.data, self.sample_rate, resample)
            self.sample_rate = resample
        
        if spectrogram:
            tf_transformer = Mag_phase_getter(fft_size=1024)
            x_mag, x_phase, x_if = tf_transformer(self.data)
            self.data = torch.stack([x_mag,x_phase])
        
#         # scale entire dataset to between 1 and -1
#         scale_factor = max(torch.max(torch.abs(self.x)) , torch.max(torch.abs(self.y)))
#         self.x = self.x/scale_factor
#         self.y = self.y/scale_factor
        
#         self.sample_rate = fs

#         if resample!=None:
#             self.x = resample_waveform(self.x, self.sample_rate, resample)
#             self.y = resample_waveform(self.y, self.sample_rate, resample)
#             self.sample_rate = resample
        
#         if spectrogram:
#             tf_transformer = Mag_phase_getter(fft_size=1024)
#             x_mag, x_phase, x_if = tf_transformer(self.x)
#             self.x = torch.stack([x_mag,x_phase])
#             y_mag, y_phase, y_if = tf_transformer(self.y)
#             self.y = torch.stack([y_mag,y_phase])

        self.pairs = None
        self._preprocess(train)

    def _preprocess(self, train):
        pairs = self._make_pairs()
        train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=1, shuffle=True)
        self.pairs = train_pairs if train else test_pairs  
        
    def __len__(self):
        return len(self.pairs)   
    
    def __getitem__(self, idx):
        return self.pairs[idx]

    def _make_pairs(self):
        if self.spectrogram:
            cutoff_idx = round(self.cutoff*self.sample_rate) 
            x, y = splot
            y = 
        else:
            cutoff_idx = round(self.cutoff*self.sample_rate) 
            x = self.data[:,:cutoff_idx]
            y = self.data[:,cutoff_idx:]
#         if self.x.dim()==2:
#             self.x = self.x.view(self.x.shape[0],-1,self.x.shape[1])
#             self.y = self.y.view(self.y.shape[0],-1,self.y.shape[1])
        return list(zip(x, y))
    

    
class Mag_phase_getter():
    def __init__(self, fft_size=1024):
        self.stft = T.Spectrogram(n_fft=fft_size, power=None, hop_length=n_fft//4)
        
    def transform(self, x):
        x_stft = self.stft(x)
        
        x_mag = torch.abs(x_stft)
        x_phase = torch.angle(x_stft)
        x_if = self._get_if(x_phase)
        return x_mag, x_phase, x_if
    
    def _get_instantaneous_frequency(self, arg):
        unwrapped_angle = np.unwrap(arg).astype(np.single)
        return np.concatenate([unwrapped_angle[:,:,0:1], np.diff(unwrapped_angle, n=1)], axis=-1)

    
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