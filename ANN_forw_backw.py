import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

#create Data class
class Data(Dataset):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.train_numbers = x_train
        self.train_labels = y_train

        self.test_numbers = x_test
        self.test_labels = y_test

        self.len = len(self.train_labels)

    def __getitem__(self, item):
        return self.train_numbers[item], self.train_labels[item]

    def __len__(self):
        return self.len


#this will be the class for our forward ANN
class ForwardANN(nn.Module):
    def __init__(self, numIns, numOuts):
        super(ForwardANN, self).__init__()

        #to make reversing easier later, only use linear layers.
        #going from nimIns to 80
        self.in_to_one = nn.Linear(numIns, 80)
        #then from 80 to 55
        self.one_to_two = nn.Linear(80, 55)
        #then from 55 to 25
        self.two_to_three = nn.Linear(55, 25)
        #lastly from 25 to numOuts
        self.three_to_out = nn.Linear(25, numOuts)

    def forward(self, x):
        #flatten image
        x = torch.flatten(x, 1)
        #run x through relu for all layers
        x = F.relu(self.in_to_one(x))
        x = F.relu(self.one_to_two(x))
        x = F.relu(self.two_to_three(x))
        x = F.relu(self.three_to_out(x))
        return x

#train forward ANN
def trainANN(dataset, numIns, numOuts, epochs=5, batch_size=25, lr=0.001):

    # create data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    # create ANN
    FANN = ForwardANN(numIns, numOuts)
    print(f"Total parameters: {sum(param.numel() for param in FANN.parameters())}")

    # loss function
    cross_entropy = nn.CrossEntropyLoss()

    # Use Adam Optimizer
    optimizer = torch.optim.Adam(FANN.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        for _, data in enumerate(tqdm(data_loader)):
            x, y = data

            optimizer.zero_grad()

            output = FANN(x)

            loss = cross_entropy(output, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        print(f"Running loss for epoch {epoch + 1}: {running_loss:.4f}")
        running_loss = 0.0
        with torch.no_grad():
            predictions = torch.argmax(FANN(dataset.test_numbers), dim=1)  # Get the prediction
            correct = (predictions == dataset.test_labels).sum().item()
            print(f"Accuracy on test set: {correct / len(dataset.test_labels):.4f}")
    return FANN

#https://www.kaggle.com/datasets/vimpigro/handwritten-mongolian-cyrillic-characters-database/data?select=HMCC+letters+merged.csv
path = "files"
#used image.py to break up letters.csv into 2 files
df = pd.read_csv(path + "/letter_train.csv")
xTrain = torch.tensor(df.iloc[:, 1:].to_numpy() / 255.0, dtype=torch.float).view(-1, 1, 28, 28)
yTrain = torch.tensor(df.iloc[:, 0].to_numpy())

df = pd.read_csv(path + "/letter_test.csv")
xTest = torch.tensor(df.iloc[:, 1:].to_numpy() / 255.0, dtype=torch.float).view(-1, 1, 28, 28)
yTest = torch.tensor(df.iloc[:, 0].to_numpy())

d = Data(xTrain, yTrain, xTest, yTest)

#save trained ANN
fann = trainANN(d, 28**2, 35)


#now that we have our trained ANN, it's time to reverse it.
class BackwardANN(nn.Module):
    def __init__(self, numIns, numOuts, forANN):
        super().__init__()
        #save so that we have the number of inputs to use for later
        self.ins = int(np.sqrt(numIns))

        #layers should match the trained ANN but reversed
        self.in_to_one = nn.Linear(numOuts, 25)
        self.one_to_two = nn.Linear(25, 55)
        self.two_to_three = nn.Linear(55, 80)
        self.three_to_out = nn.Linear(80, numIns)

        #copy weights from trained ANN into backward ANN
        self.in_to_one.weight.data.copy_(forANN.three_to_out.weight.data.T)
        self.one_to_two.weight.data.copy_(forANN.two_to_three.weight.data.T)
        self.two_to_three.weight.data.copy_(forANN.one_to_two.weight.data.T)
        self.three_to_out.weight.data.copy_(forANN.in_to_one.weight.data.T)

    def forward(self, x):
        #flatten is not needed for this one
        #still running all layers through relu
        x = F.relu(self.in_to_one(x))
        x = F.relu(self.one_to_two(x))
        x = F.relu(self.two_to_three(x))
        x = F.relu(self.three_to_out(x))
        #reshaping output as we now want an image back instead of a 1D tensor
        return x.view(-1, 1, self.ins, self.ins)

#create backward ANN
bann = BackwardANN(28*28, 35, fann)

#array of characters
characters = ['А','Б','В','Г','Д','Е','Ё','Ж','З','И','Й','К','Л','М','Н','О','Ө'
    ,'П','Р','С','Т','У','Ү','Ф','Х','Ц','Ч','Ш','Щ','Ъ','Ь','Э','Ю','Я']

#loop through the letters to see what the ANN thinks they look like
start = 10
end = 15
for i in range(start, end):
    input = torch.zeros(1, 35)
    input[0, i] = 1.0
    image = bann(input).detach().squeeze().numpy()
    kernel = np.ones((9, 9), np.float32) / 81
    image = cv2.filter2D(image, -1, kernel)
    #uncomment this to see a less blurry image
    # for x in range(28):
    #     for y in range(28):
    #         if image[x,y] < .005:
    #             image[x,y] = 0
    #         elif image[x,y] < .2:
    #             image[x,y] = .5
    #         else:
    #             image[x,y] = 1
    plt.imshow(image, cmap='grey')
    plt.title(characters[i])
    plt.show()
