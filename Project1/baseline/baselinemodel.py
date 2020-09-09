import torch
import torch.multiprocessing as mp
import torch.nn.functional
import torchvision.models as models
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


if __name__ == "__main__":
    train_model_multithreaded("./gpu_model")