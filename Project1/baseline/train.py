import torch
import torch.multiprocessing as mp
import torch.nn.functional
import joblib
import numpy as np
import torchvision.models as models
from datasets import CIFAR100_Training

def train_process(rank, path, model, iterations, device):
    dataset = CIFAR100_Training(path)
    data_generator = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=1)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for itr in range(iterations):
        correct = 0
        count = 0
        running_loss = 0
        model.train()

        for x, y in data_generator:
            x = x.squeeze().to(device)
            y = y.squeeze().to(device)

            optimizer.zero_grad() # model.zero_grad() ?
            y_pred = model(x)

            loss = torch.nn.functional.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()

            count += x.size()[0]
            running_loss += loss.item()

            for i in range(x.size()[0]):
                if (y_pred[i].argmax().item() == y[i].item()):
                    correct += 1

            if (count%2000 == 0):
                print("[%2d" % rank,
                    " %2d" % itr,
                    " %8d]" % count,
                    "  ITEM LOSS: %6f" % loss.item(), 
                    "  ACC: %5f" % (correct*100.0/count),
                    "  RUNNING LOSS: %6f" % (running_loss / (count+1)))

        print("[%2d" % rank,
            " %2d" % itr,
            " %8d]" % count,
            " COMPLETED ITR ", itr)