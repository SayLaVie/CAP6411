from baselinemodel import build_model
from basisModel import basisModel
from datasets import CIFAR100_Training
import torch
import time


def eval_model(path, compression_factor=0):
    device = torch.device("cuda")
    model = build_model().to(device)
    model.load_state_dict(torch.load(path))

    if (compression_factor != 0):
        compressed_model = basisModel(model, True, True, True)
        compressed_model.update_channels(compression_factor)
        model = compressed_model.to(device)

    # 100x500x64x64x3 Data size (Classes, images, image dimensions)
    dataset = CIFAR100_Training("./validation")
    data_generator = torch.utils.data.DataLoader(dataset, shuffle=True)

    correct = 0
    count = 0
    running_loss = 0
    start = time.perf_counter()
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
    eval_model("./models/gpu_model", 0.8)
