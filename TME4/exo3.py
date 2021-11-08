from utils import RNN, device,  ForecastMetroDataset
from torch.utils.tensorboard import SummaryWriter
import datetime

from torch.utils.data import  DataLoader
import torch
import torch.nn as nn

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 1
#Taille du batch
BATCH_SIZE = 32

writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

PATH = "data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

pred_nb = 2
DIM_HIDDEN = 10


max_iter = 51
lr = 0.001

rnn = RNN(DIM_INPUT, DIM_HIDDEN, DIM_INPUT).to(device)

def train_prediction(rnn,data_train,data_test,lr,max_iter):

    loss = nn.MSELoss()
    optim = torch.optim.AdamW(rnn.parameters(), lr)

    print("Training for {} epochs".format(max_iter))

    for iter in range(max_iter):
        losstrain = []

        for x,y in data_train:
            try:
              x = x.view(BATCH_SIZE,-1,DIM_INPUT).to(device)
              y = y.view(BATCH_SIZE,-1,DIM_INPUT).to(device)
            except RuntimeError:
              continue
            yhat = rnn(x)
            decoded = torch.stack([rnn.decode(i) for i in yhat],1)

            l = loss(decoded,y)
            l.backward()
            optim.step()
            optim.zero_grad()


            losstrain.append(l)


        losstest = []


        for x,y in data_test:
            try:
              x = x.view(BATCH_SIZE,-1,DIM_INPUT).to(device)
              y = y.view(BATCH_SIZE,-1,DIM_INPUT).to(device)
            except RuntimeError:
              continue

            with torch.no_grad():

                yhat = rnn(x[:,:-pred_nb*CLASSES,:])
                decoded = [rnn.decode(i) for i in yhat]
                h = yhat[-1]
                for i in range(pred_nb*CLASSES):
                  h = rnn.one_step(decoded[-1],h)
                  next = rnn.decode(h)
                  decoded.append(next)
                decoded = torch.stack(decoded,1)
            l = loss(decoded,y)


            losstest.append(l)

        if iter % 10 == 0:
          print("iteration {}, loss train : {}, loss test : {}".format(iter,
                                    torch.tensor(losstrain).mean(),
                                    torch.tensor(losstest).mean()))

        writer.add_scalars('Prediction/',{'train':torch.tensor(losstrain).mean(),'test':torch.tensor(losstest).mean()}, iter)
