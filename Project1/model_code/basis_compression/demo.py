import torch
import torch.nn as nn
import copy
import math
from basisModel import basisModel,trace_model,display_stats
import torchvision.models as models
from PIL import Image
from torchvision import transforms


with open('classes.json') as f:
  labels = [line.strip().split('"')[1] for line in f.readlines()]

transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

img = Image.open("download.jpeg")
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

use_weights = True
add_bn = False
fixed_basbs = True
compression_factor = 0.8

#using the pretrained model from resnet
model = models.resnet50(pretrained=True)
compressed_model = basisModel(model, use_weights, add_bn, fixed_basbs)
compressed_model.update_channels(compression_factor)

display_stats(compressed_model, (480,480))

out_compressed = compressed_model(batch_t)

