# Utility for taking full dataset and distributing it into seperate files for concurrent loading
import torch
import numpy as np
import pickle
import joblib
import os

def distribute_data(path, data, labels, batchsize=4):
    num = min(len(data), len(labels))
    batches = int(num/batchsize)

    # Create target directory if not exists
    os.makedirs(path, exist_ok=True)

    # Create metadata file with information about number of batches and batch size
    metadata = dict()
    metadata["batches"] = batches
    metadata["batch_size"] = batchsize

    with open(str(path)+"/metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    # Create data files, 1 per batch
    for i in range(batches):
        batch = dict()
        batch_start = i * batchsize
        batch_end = (i+1) * batchsize
        batch["data"] = data[batch_start : batch_end]
        batch["labels"] = labels[batch_start : batch_end]

        with open(str(path)+"/"+str(i)+".pkl", "wb") as f:
            pickle.dump(batch, f)


if __name__ == "__main__":
    dataset = joblib.load("./resized-cifar-100-python/resized-test")
    distribute_data("./validation", dataset[b'data'], dataset[b'fine_labels'])
