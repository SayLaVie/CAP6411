import torch
import torch.multiprocessing as mp
import torch.nn.functional
import joblib
import numpy as np
import torchvision.models as models
from datasets import CIFAR100_Training
from train import train_process

def build_model():
    # VGG 16 model with output layer resized for 100 classes
    model = models.vgg16_bn(pretrained=True)
    model.classifier[6] = torch.nn.Linear(4096, 100)
    return model

# Nonfunctional: need to add logic for iterating through batch
def train_model(path):
    model = build_model()

    model.train()

    dataset = CIFAR100_Training("./data")
    data_generator = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)

    iterations = 50
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for itr in range(iterations):
        correct = 0
        count = 0
        running_loss = 0

        for x, y in data_generator:
            model.zero_grad()
            y_pred = model(x)

            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()

            count += 1
            running_loss += loss.item()

            if (y_pred.argmax().item() == y.item()):
                correct += 1

            if (count%2000 == 0):
                print("[%2d" % itr,
                    " %8d]" % count,
                    "  ITEM LOSS: %6f" % loss.item(), 
                    "  ACC: %5f" % (correct*100.0/count), 
                    "%%  Y: %5f" % y.item(), 
                    "  Y_PRED: %5f" % y_pred.argmax().item(),
                    "  RUNNING LOSS: %6f" % (running_loss / (count+1)))

        print("-----------COMPLETED ITR ", itr, "-----------")

    torch.save(model.state_dict(), path)


def train_model_multithreaded(path, workers=2):
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.manual_seed(1)
    mp.set_start_method("spawn")

    model = build_model().to(device)
    model.share_memory()

    iterations = 5

    threads = []
    for rank in range(workers):
        p = mp.Process(target=train_process, args=(rank, "./data", model, iterations, device))
        p.start()
        threads.append(p)

    for t in threads:
        t.join()

    print("ALL THREADS JOINED, SAVING MODEL TO \"", path,"\"")
    torch.save(model.state_dict(), path)


def eval_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))

    # 100x500x64x64x3 Data size (Classes, images, image dimensions)
    test_data = joblib.load("./resized-cifar-100-python/resized-test")

    images = 10000
    correct = 0

    for i in range(images):
        x = data_to_tensor(test_data[b'data'][i])
        y = label_to_tensor(test_data[b'fine_labels'][i])

        y_pred = model(x)

        if (y_pred.argmax().item() == y.item()):
            correct += 1
        
        if i%1000 == 0:
            print("IMG: ", i, "  CORRECT: ", correct, "/", i+1, "  Y: ", y, "  Y_PRED: ", y_pred.argmax())


if __name__ == "__main__":
    train_model_multithreaded("./gpu_model")