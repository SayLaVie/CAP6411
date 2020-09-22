from baselinemodel import build_model, replace_dcf
from basisModel import basisModel
from datasets import CIFAR100_Training
import torch
from torch2trt import torch2trt
import time
from DCF import *

def eval_model(path, compression_factor=0, use_dcf=false, as_trt=False):
    device = torch.device("cuda")
    model = build_model().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    if (compression_factor != 0 and use_dcf):
        replace_dcf(model)

    elif compression_factor != 0:
        compressed_model = basisModel(model, True, True, True)
        compressed_model.update_channels(compression_factor)
        compressed_model.cuda()
        model = compressed_model

    print("MODEL: ", model)

    if as_trt:
        print("converting to TRT")
        # torch.Size([4, 3, 64, 64]) is actual input
        x = torch.rand((4, 3, 224, 224)).cuda()
        # x = torch.ones((4, 3, 64, 64)).cuda() # errors out
        model_trt = torch2trt(model, [x], max_batch_size=4)
        model = model_trt.to(device)
        print("finished converting to TRT")

    # 100x500x64x64x3 Data size (Classes, images, image dimensions)
    dataset = CIFAR100_Training("./validation")
    data_generator = torch.utils.data.DataLoader(dataset, shuffle=True)

    correct = 0
    count = 0
    running_loss = 0
    start = time.perf_counter()

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

            if (count % 2000 == 0):
                print("[%8d]" % count,
                      "  ACC: %5f" % (correct*100.0/count),
                      "  LOSS: %6f" % (running_loss / (count+1)))

        stop = time.perf_counter()
        runtime = stop - start
        print("[%8d]" % count,
            "  ACC: %5f" % (correct*100.0/count),
            "  LOSS: %6f" % (running_loss / (count+1)))

        print("Total inference time: %5f seconds" % runtime,
            "\nFrames per second: %5f" % (count/runtime))


# TODO: parameterize model to evaluate
if __name__ == "__main__":
    # eval_model("./models/gpu_model", as_trt=True)
    eval_model("./models/gpu_model")