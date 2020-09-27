from baselinemodel import build_model# , replace_dcf
from basisModel import basisModel
from datasets import CIFAR100_Training
# from DCF import *
import subprocess
import sys
import torch
import torch.autograd.profiler as profiler
from torch2trt import torch2trt

model_path = "./models/gpu_model"
dataset_path = "./validation"
device = torch.device("cuda")
compression_factor = 0
as_trt = True
trt_fp_16_mode = True
use_dcf = False
num_iterations = 1
run_tegra = False


def eval():
    model = setup_model()
    run_infer(model)


def setup_model():
    global device, model_path, compression_factor, as_trt

    model = build_model().to(device)

    # if use_dcf:
    #     replace_dcf(model)

    model.load_state_dict(torch.load(model_path))

    if (compression_factor != 0):
        compressed_model = basisModel(model, True, True, True)
        compressed_model.update_channels(compression_factor)
        model = compressed_model.to(device)

    model.eval()

    if as_trt:
        print("converting to TRT")
        # torch.Size([4, 3, 64, 64]) is actual input
        # x = torch.rand((4, 3, 224, 224)).cuda()
        x = torch.ones((4, 3, 64, 64)).cuda()  # errors out
        model_trt = torch2trt(model, [x], max_batch_size=4, fp16_mode=trt_fp_16_mode)
        model = model_trt.to(device)
        print("finished converting to TRT")

    return model


def run_infer(model):
    global dataset_path, device, run_tegra

    # 100x500x64x64x3 Data size (Classes, images, image dimensions)
    dataset = CIFAR100_Training(dataset_path)
    data_generator = torch.utils.data.DataLoader(dataset, shuffle=True)

    # Warm-up
    with torch.no_grad():
        for x, y in data_generator:
            x = x.squeeze().to(device)
            model(x)

    for i in range(num_iterations):
        print("Iteration: ", i)
        print("Iteration: ", i, file=sys.stderr)
        
        correct = 0
        count = 0
        running_loss = 0
        time_list = []
        if run_tegra:
            start_tegrastats()

        with torch.no_grad():
            for x, y in data_generator:
                x = x.squeeze().to(device)
                y = y.squeeze().to(device)

                # Only time the actual inferences
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                y_pred = model(x)
                torch.cuda.synchronize()
                end.record()
                torch.cuda.synchronize()

                time_list.append(start.elapsed_time(end))

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

            if run_tegra:
                stop_tegrastats()
            
            time_sum = sum(time_list) / 1000  # convert milliseconds to seconds
            time_per_image = time_sum / len(time_list) / 4  # batch size = 4

            print("Total inference time: %5f seconds" % time_sum,
                "\nTime per image: %5f seconds" % time_per_image,
                "\nImages per second: %5f" % (count / time_sum))
            
            profile_model(model)


def start_tegrastats():
    subprocess.Popen(
        "sshpass -p {passwd} ssh {user}@{host} {cmd}".format(
            passwd='cv-systems', user='dev', host='192.168.1.3',
            cmd='/home/dev/project-git/Project1/scripts/start-tegrastats.sh'
            ), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).communicate()


def stop_tegrastats():
    subprocess.Popen(
        "sshpass -p {passwd} ssh {user}@{host} {cmd}".format(
            passwd='cv-systems', user='dev', host='192.168.1.3',
            cmd='/home/dev/project-git/Project1/scripts/stop-tegrastats.sh'
            ), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).communicate()


def profile_model(model):
    x = torch.randn((4, 3, 64, 64)).cuda()

    with profiler.profile(record_shapes=True, profile_memory=True, use_cuda=True) as prof:
        with profiler.record_function("model_inference"):
            model(x)

    print(prof.key_averages().table(sort_by="cpu_time_total"))


# TODO: parameterize model to evaluate
if __name__ == "__main__":
    eval()
