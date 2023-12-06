import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '../classify')
from classify import compute_metrics

projectSampleRate = 16000
hop_len = 256
num_mfccs = 13
spec_frame_rate = np.floor((1/hop_len) * projectSampleRate)
target_length = 10

batchSize = 16
learningRate = 0.01
weight_decay = 0.0001
epoch = 100
# MLP implementation
device = torch.device('mps')

class mtload(Dataset):
    def __init__(self, csv_file, target_length, sample_rate_mfcc):
        """
        Args:
            csv_file (string): Path to the csv file with annotations and file paths.
            target_length (int): Desxired length of the spectrograms in seconds.
            sample_rate (int): Sample rate of the spectrograms.
        """
        self.annotations = pd.read_csv(csv_file)
        self.target_length = target_length
        self.sample_rate_mfcc = sample_rate_mfcc

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        mfcc_path = self.annotations.iloc[idx]['mfcc_filepath']
        mfcc = np.load(f"../{mfcc_path}")

        intox = self.annotations.iloc[idx]['alc_mapped']
        if intox == "Intoxicated":
            target = 1.0

        else:
            target = 0.0

        mfcc_tensor = torch.from_numpy(mfcc).type(torch.float32)
        mfcc_tensor = torch.swapaxes(mfcc_tensor, 0, 1)
        mfcc_tensor = torch.flatten(mfcc_tensor)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return mfcc_tensor, target_tensor



def create_data_loader(csv_file, batch_size, target_length, sample_rate_mfcc):
    dataset = mtload(csv_file=csv_file, target_length=target_length, sample_rate_mfcc=sample_rate_mfcc)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example usage
train_data_loader = create_data_loader('../train_clipped_mfcc.csv',
                                       batch_size=batchSize,
                                       target_length = target_length,
                                       sample_rate_mfcc = spec_frame_rate)

test_data_loader = create_data_loader('../test_clipped_mfcc.csv',
                                      batch_size=1,
                                      target_length=target_length,
                                      sample_rate_mfcc=spec_frame_rate)

print(len(train_data_loader))
print(len(test_data_loader))

class MLP(nn.Module):
    def __init__(self, input_size, class_output):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.nl = nn.Tanh()
        self.fc2 = nn.Linear(10, class_output)
    def forward(self, x):
        out = self.fc1(x)
        out = self.nl(out)
        out = self.fc2(out)
        return out

train_losses = []
def train(model, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_data_loader):
        data = data.to(device)
        target = target.to(device)

        model.zero_grad()
        output = model(data)
        output = torch.squeeze(output)
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_data_loader)
    train_losses.append(train_loss)

    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_data_loader):
            data = data.to(device)
            target = target.to(device)

            model.zero_grad()
            output = model(data)

            if output.float() > 0:
                pred = 1
            else:
                pred = 0

            if pred == target.float():
                correct += 1
        accuracy = 100. * correct / len(test_data_loader.dataset)

    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.2f}, Test Loss: {accuracy:.2f}')


def evaluate(model):
    model.eval()
    predss = []
    groundt = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_data_loader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            for output_instance in output:
                if output_instance.float() > 0:
                    pred = 1
                else:
                    pred = 0
                predss.append(pred)

            for target_instance in target:
                groundt.append(target_instance.cpu().numpy())


    train_accuracy_sanna = compute_metrics(groundt, predss)
    print(f'Training Accuracy: {train_accuracy_sanna}')

    predss = []
    groundt = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_data_loader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            if output.float() > 0:
                pred = 1
            else:
                pred = 0

            predss.append(pred)
            groundt.append(target.cpu().numpy())

    test_accuracy_sanna = compute_metrics(groundt, predss)
    print(f'Testing Accuracy: {test_accuracy_sanna}')






input_size = num_mfccs*target_length*spec_frame_rate
input_size = int(input_size)
mlp = MLP(input_size=input_size,
            class_output=1)
mlp.to(device)

num_LSTM_parameters = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
print(f'Number of parameters in the model: {num_LSTM_parameters}')

optimizer = optim.Adam(mlp.parameters(), lr=learningRate, weight_decay=weight_decay)
criterion = nn.BCEWithLogitsLoss()

log_interval = 1
for epoch in range(1, epoch):
    # scheduler.step()
    train(mlp, epoch)
evaluate(mlp)

