import math
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

import os
class MFCC_preprocessor():

    """
    Loads, preprocesses, and writes audio data for modeling.
    """
    def __init__(
        self,
        proj_fs: int = 16000,
        num_mels: int = 64,
        n_mfcc: int = 64,
        n_fft: int = 512,
        hop_length: int = 256,
        target_length: float = 10
    ):
        self.proj_fs = proj_fs
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_mels = num_mels
        self.target_length = target_length

    def processMFCC(self, inputFilePath, normalize=True):
        waveform, sample_rate = torchaudio.load(inputFilePath, normalize=normalize)
        transform = T.MFCC(sample_rate = self.proj_fs,
                            n_mfcc = self.n_mfcc,
                            melkwargs = {"n_fft": self.n_fft, "hop_length": self.hop_length, "n_mels": self.num_mels, "center": False})
        MFCC = transform(waveform).numpy()
        spec_frame_rate = np.floor((1/self.hop_length) * self.proj_fs)

        # Calculate the number of samples for target length
        num_samples = (self.target_length * spec_frame_rate).astype(int)
        MFCC = np.squeeze(MFCC)
        # Pad or cut the spectrogram
        if MFCC.shape[1] > num_samples:
            # Cut the spectrogram
            MFCC = MFCC[:, :num_samples]

        elif MFCC.shape[1] < num_samples:
            # Pad the spectrogram with zeros
            padding_size = num_samples - MFCC.shape[1]
            MFCC = np.pad(MFCC, ((0, 0), (0, padding_size)), mode='constant')
        return MFCC

    def plotMFCC(self, mfcc, ax=None):
        ylabel = "MFCC"
        xlabel = "Frame"
        title = "MFCC over time frames"
        if ax is None:
            _, ax = plt.subplots(1, 1)
        if title is not None:
            ax.set_title(title)

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        ax.set_title('MFCC')
        ax.imshow(mfcc, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
        plt.show()

    def saveMFCC(self, mfcc, file_path, save_dir='mfccs'):

        # Create save directory if it doesn't exist
        if not os.path.exists(f"../{save_dir}"):
            os.makedirs(f"../{save_dir}")

        # save as np array
        filename = f"{save_dir}/{os.path.splitext(os.path.basename(file_path))[0]}.npy"
        np.save(f"../{filename}", mfcc)

        return filename