from utils import RNN, device,SampleMetroDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime


# Nombre de stations utilisé
CLASSES = 10
# Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
# Taille du batch
BATCH_SIZE = 64


writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

PATH = "data/"

matrix_train, matrix_test = torch.load(open(PATH + "hzdataset.pch", "rb"))
# print(train.shape) #18,73,80,2
# print(test.shape) #7,73,80,2

ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH,
                             stations_max=ds_train.stations_max)

data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)



DIM_HIDDEN = 10


max_iter = 1001
lr = 0.001

rnn = RNN(DIM_INPUT, DIM_HIDDEN, CLASSES).to(device)

def train_classification(rnn,data_train,data_test,lr,max_iter):

    print("Training for {} epochs".format(max_iter))


    loss = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(rnn.parameters(),lr)

    for iter in range(max_iter):

        losstrain = []
        acctrain = []

        for x,y in data_train:
            x = x.to(device)
            y = y.to(device)
            yhat = rnn(x)
            decoded = rnn.decode(yhat[-1])
            l = loss(decoded,y)
            l.backward()
            optim.step()
            optim.zero_grad()

            acc = sum((decoded.argmax(1) == y.reshape(-1))).item() / y.shape[0]
            losstrain.append(l)
            acctrain.append(acc)


        losstest = []
        acctest = []


        for x,y in data_test:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():

                yhat = rnn(x)
                decoded = rnn.decode(yhat[-1])
                l = loss(decoded,y)


                acc = sum((decoded.argmax(1) == y.reshape(-1))).item() / y.shape[0]
                losstest.append(l)
                acctest.append(acc)

        if iter % 10 == 0:
          print("iteration {}, loss train : {}, acc train :{}, loss test : {}, acc test : {}".format(iter,
                                    torch.tensor(losstrain).mean(),torch.tensor(acctrain).mean(),
                                    torch.tensor(losstest).mean(),torch.tensor(acctest).mean()))


        writer.add_scalars('Prediction/', {'train': torch.tensor(losstrain).mean(), 'test': torch.tensor(losstest).mean()},
                   iter)

        writer.add_scalars('Prediction_acc/', {'train': torch.tensor(acctrain).mean(), 'test': torch.tensor(acctest).mean()},
                   iter)
