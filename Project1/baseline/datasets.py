import numpy as np
import pickle
import torch


class CIFAR100_Training(torch.utils.data.Dataset):
    def data_to_tensor(self, data):
        formatted_data = np.asarray(data)
        formatted_data = formatted_data.reshape(len(formatted_data), 3, 64, 64)/255.0
        return torch.from_numpy(formatted_data).float()

    def label_to_tensor(self, label):
        return torch.from_numpy(np.asarray(label)).long()

    def load_items(self):
        # TODO: better performance with a contiguous array instead of list?
        self.items = []
        for index in range(self.length):
            batch_file = open(str(self.path)+"/"+str(index)+".pkl", "rb")
            batch = pickle.load(batch_file)
            x = self.data_to_tensor(batch["data"])
            y = self.label_to_tensor(batch["labels"])
            self.items.append((x, y))

    def __init__(self, path):
        metadata_file = open(str(path)+"/metadata.pkl", "rb")
        metadata = pickle.load(metadata_file)
        self.length = metadata["batches"]
        self.path = path
        self.load_items()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.items[index]