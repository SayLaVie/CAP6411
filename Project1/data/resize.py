import argparse
import cv2
import joblib
import numpy as np
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="resize CIFAR")
    parser.add_argument('--input', '-i', type=str, required=True, help='input pickle to resize')
    parser.add_argument('--output', '-o', type=str, required=True, help='output archive (joblib) after resizing')

    return parser.parse_args()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def resize(data):
    print("Old shape: ", data.shape)

    images = np.asarray(data.reshape((len(data), 3, 32, 32)))
    new_images = np.empty((len(data), 12288))

    for idx_outer, image in enumerate(images):
        new_bands = np.empty((3, 64, 64))
        for idx_inner, band in enumerate(image):
            new_bands[idx_inner] = cv2.resize(band, (64, 64))            
        new_images[idx_outer] = new_bands.flatten()

    print("New shape: ", new_images.shape)
    return new_images
    
if __name__ == '__main__':
   args = parse_args()

   batch = unpickle(args.input)
   print(batch.keys())

   data = batch[b'data']
   # filenames = batch[b'filenames']
   # batch_label = batch[b'batch_label']
   # fine_labels = batch[b'fine_labels']
   # course_labels = batch[b'coarse_labels']

   new_images = resize(data)
   batch[b'data'] = new_images

   joblib.dump(batch, args.output)
