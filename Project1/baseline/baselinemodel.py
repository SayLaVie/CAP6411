import torch
import torch.multiprocessing as mp
import torch.nn.functional
import torchvision.models as models
from train import train_process
from DCF import *

def replace_dcf(module):
    for a, b in list(module.named_children()):
        for n, m in list(b.named_children()):
            if isinstance(m, nn.Conv2d):
                b._modules[n] = Conv_DCF(m.in_channels, m.out_channels, m.kernel_size[0], stride=m.stride[0], padding=m.padding[0]).cuda()

def build_model():
    # VGG 16 model with output layer resized for 100 classes
    model = models.vgg16_bn(pretrained=True)
    model.classifier[6] = torch.nn.Linear(4096, 100)
    return model


def train_model_multithreaded(path, workers=2, use_dcf_layers=False):
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.manual_seed(1)
    mp.set_start_method("spawn")

    model = build_model().to(device)

    if (use_dcf_layers):
        replace_dcf(model)

    model.share_memory()

    print("Trainning model: ", model)

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
    train_model_multithreaded("./dcf_model", use_dcf_layers=True)