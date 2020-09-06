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
    device = torch.device("cuda")
    model = build_model().to(device)
    model.load_state_dict(torch.load(path))

    # 100x500x64x64x3 Data size (Classes, images, image dimensions)
    dataset = CIFAR100_Training("./validation")
    data_generator = torch.utils.data.DataLoader(dataset, shuffle=True)


    correct = 0
    count = 0
    running_loss = 0
    model.eval()

    with torch.no_grad():
        for x, y in data_generator:
            x = x.squeeze().to(device)
            y = y.squeeze().to(device)

            y_pred = model(x)

            loss = torch.nn.functional.cross_entropy(y_pred, y)

            count += x.size()[0]
            running_loss += loss.item()

            for i in range(x.size()[0]):
                if (y_pred[i].argmax().item() == y[i].item()):
                    correct += 1


            if (count%2000 == 0):
                print("[%8d]" % count,
                    "  ACC: %5f" % (correct*100.0/count),
                    "  LOSS: %6f" % (running_loss / (count+1)))

    print("[%8d]" % count,
        "  ACC: %5f" % (correct*100.0/count),
        "  LOSS: %6f" % (running_loss / (count+1)))


if __name__ == "__main__":
    train_model_multithreaded("./gpu_model")
    eval_model("./gpu_model")