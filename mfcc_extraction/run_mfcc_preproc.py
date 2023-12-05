import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

import os

import torchaudio.transforms as T
import torchaudio.functional as F
import mfcc_preproc
import torchaudio
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from alive_progress import alive_bar


# INTERFACE
projectSampleRate = 16000
fft_len = 512
hop_len = 256
num_mels = 64
num_mfccs = 13
target_length = 10.0
calculateMFCCs = True

preproc = mfcc_preproc.MFCC_preprocessor(
    proj_fs=projectSampleRate,
    num_mels=num_mels,
    n_mfcc=num_mfccs,
    n_fft=fft_len,
    hop_length=hop_len,
    target_length=target_length)

# Sanity Check
mfcc = preproc.processMFCC("my_speech.wav")
preproc.plotMFCC(mfcc)
print(mfcc.shape)

if(calculateMFCCs):
    # Read train and test CSVs
    train_df = pd.read_csv('../train_clipped.csv')
    test_df = pd.read_csv('../test_clipped.csv')

    # Calculate and save MFCCs of Train data
    filepaths = train_df['resampled_path'].tolist()
    mfcc_filepaths = []
    print("Calculate and save MFCCs of Train data...")
    with alive_bar(len(filepaths), force_tty=True) as bar:
        for filepath in filepaths:
            filepath = f"../general/{filepath}"
            mfcc = preproc.processMFCC(filepath)
            mfcc_filepath = preproc.saveMFCC(mfcc, filepath)
            mfcc_filepaths.append(mfcc_filepath)
            bar()
    train_df['mfcc_filepath'] = mfcc_filepaths
    train_df.to_csv('../train_clipped_mfcc.csv', index=False)

    # Calculate and save MFCCs of Test data
    filepaths = test_df['resampled_path'].tolist()
    mfcc_filepaths = []
    print("Calculate and save MFCCs of Test data...")
    with alive_bar(len(filepaths), force_tty=True) as bar:
        for filepath in filepaths:
            filepath = f"../general/{filepath}"
            mfcc = preproc.processMFCC(filepath)
            mfcc_filepath = preproc.saveMFCC(mfcc, filepath)
            mfcc_filepaths.append(mfcc_filepath)
            bar()
    test_df['mfcc_filepath'] = mfcc_filepaths
    test_df.to_csv('../test_clipped_mfcc.csv', index=False)




