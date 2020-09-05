import torch
import joblib
import numpy as np
import torchvision.models as models
import torchvision.datasets as datasets

def data_to_tensor(data):
    arry = np.zeros((1, 3, 64, 64)).astype(float)

    for a in range(3):
        for b in range(64):
            for c in range(64):
                arry[0][a][b][c] = data[a*64*64 + b*64 + c]

    return torch.from_numpy(arry).float().cuda()

def label_to_tensor(label):
    arry = np.zeros((1))
    arry[0] = label

    return torch.from_numpy(arry).long().cuda()

def build_model():
    # VGG 16 model with output layer resized for 100 classes
    model = models.vgg16_bn()
    model.classifier[6] = torch.nn.Linear(4096, 100)
    model.cuda()
    return model

def train_model(path):
    model = build_model()
    model.train()

    # 100x500x64x64x3 Data size (Classes, images, image dimensions)
    train_data = joblib.load("./resized-cifar-100-python/resized-train")

    learning_rate = 1e-4
    iterations = 3
    images = 50000
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for itr in range(iterations):

        correct = 0
        count = 0
        for i in range(images):
            x = data_to_tensor(train_data[b'data'][i])
            y = label_to_tensor(train_data[b'fine_labels'][i])

            model.zero_grad()
            y_pred = model(x)

            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()

            count += 1
            if (y_pred.argmax().item() == y.item()):
                correct += 1
                print("IMG: ", i, 
                    "\n   LOSS: ", loss, 
                    "\n   ACC: ", (correct*100.0/count), 
                    "%\n   Y: ", y.item(), 
                    "\n   Y_PRED_MAX: ", y_pred.argmax().item(),
                    "\n   Y_PRED: ", y_pred.data,
                    "\n\n")

            if (i%10000 == 0):
                print("IMG: ", i, 
                    "\n   LOSS: ", loss, 
                    "\n   ACC: ", (correct*100.0/count), 
                    "%\n   Y: ", y.item(), 
                    "\n   Y_PRED_MAX: ", y_pred.argmax().item(),
                    "\n   Y_PRED: ", y_pred.data,
                    "\n\n")
            
            if (i == images-1):
                print("-----------COMPLETED ITR ", itr, "-----------")


            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad

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


train_model("./model")