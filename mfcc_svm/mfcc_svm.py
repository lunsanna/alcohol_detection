import numpy as np
import pandas as pd

import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, '../classify')
from classify import compute_metrics

df = pd.read_csv('../train_clipped_mfcc.csv')

# get all the unique values of the column 'alc'
# calculate the percentage of rows with 'alc' column value as 'a'
a_rows = df[df['alc'] == 'a']
na_rows = df[df['alc'] == 'na']

print("percentage of na rows: ", len(na_rows)/(len(a_rows)+len(na_rows)))

# Load CSV and extract bottleneck features and labels
def load_features_and_labels(csv_path):
    df = pd.read_csv(csv_path)

    # drop the rows with 'cna' in the alc column
    df = df[df['alc'] != 'cna']

    print(df.shape)

    features = [np.load('../' + fp) for fp in df['mfcc_filepath']]
    labels = df['alc'].map({'na': 0, 'a': 1}).values

    return np.array(features), labels

# Load the data
train_features, train_labels = load_features_and_labels('../train_clipped_mfcc.csv')
test_features, test_labels = load_features_and_labels('../test_clipped_mfcc.csv')

print(train_features.shape, train_labels.shape)

# Create and train the SVM
clf = svm.SVC()

# Flatten MFCCs
train_features_flat = []
for entry in train_features:
    # entry = np.ndarray.flatten(entry, order='C')
    entry = np.ndarray.flatten(entry, order='F')
    train_features_flat.append(entry)

test_features_flat = []
for entry in test_features:
    # entry = np.ndarray.flatten(entry, order='C')
    entry = np.ndarray.flatten(entry, order='F')
    test_features_flat.append(entry)

clf.fit(train_features_flat, train_labels)

# Predict on the training set
train_predictions = clf.predict(train_features_flat)
train_accuracy = accuracy_score(train_labels, train_predictions)
train_accuracy_sanna = compute_metrics(train_labels, train_predictions)


# Predict on the testing set
test_predictions = clf.predict(test_features_flat)
test_accuracy = accuracy_score(test_labels, test_predictions)
test_accuracy_sanna = compute_metrics(test_labels, test_predictions)

# Print the accuracies
print(f'Training Accuracy: {train_accuracy_sanna}')
print(f'Testing Accuracy: {test_accuracy_sanna}')