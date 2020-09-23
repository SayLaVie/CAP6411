from baselinemodel import build_model, replace_dcf
from basisModel import basisModel
from datasets import CIFAR100_Training
import torch
import time
from DCF import *

def eval_model(path, compression_factor=0, use_dcf=False, as_trt=False):
    device = torch.device("cuda")
    model = build_model().to(device)

    if (use_dcf):
        replace_dcf(model)

    model.load_state_dict(torch.load(path))

    if (compression_factor != 0 and not use_dcf):
        compressed_model = basisModel(model, True, True, True)
        compressed_model.update_channels(compression_factor)
        compressed_model.cuda()
        model = compressed_model

    model.eval()
    print("MODEL: ", model)
    
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
    # eval_model("./models/gpu_model")
    eval_model("./models/dcf_model20", use_dcf=True)
