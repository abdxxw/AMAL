from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime


#######################DATA LOADER##############################

class DataMNIST(Dataset):

    def __init__(self, X, Y):

        self.exemples = torch.from_numpy(X.reshape(X.shape[0],-1) / 255.0)
        self.labels = torch.from_numpy(Y.copy()).float()
        self.len = self.exemples.shape[0]

    def __getitem__(self, index):
        return self.exemples[index], self.labels[index]

    def __len__(self):
        return self.len


# Téléchargement des données


from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()
# print(train_images.shape)
# print(test_images.shape)


batch_size = 64
train = DataLoader(DataMNIST(train_images,train_labels),shuffle=True,batch_size=batch_size)
test = DataLoader(DataMNIST(test_images,test_labels),shuffle=True,batch_size=batch_size)

# for x,y in train:
#     print(x.shape)
#     break


####################### GPU ##############################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####################### Checkpoints ##############################

class State(object) :
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0, 0


####################### Auto encodeur ##############################

class AutoEncoder(torch.nn.Module):
    def __init__(self, dim_in, dim_latent):

        super(AutoEncoder, self).__init__()

        # Encode
        self.en_layer = torch.nn.Linear(dim_in, dim_latent)
        self.en_act = torch.nn.ReLU()
        # Decode
        self.dec_act = torch.nn.Sigmoid()

    def encode(self, data):
        return self.en_act(self.en_layer(data))

    def decode(self, encoded):
        return self.dec_act(torch.nn.functional.linear(encoded, self.en_layer.weight.t(), bias=None))

    def forward(self, data):
        return self.decode(self.encode(data))


# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
# images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# # Permet de fabriquer une grille d'images
# images = make_grid(images)
# # Affichage avec tensorboard
# writer.add_image(f'samples', images, 0)
#
#
# savepath = Path("model.pch")

#  TODO:

dim = 128
max_iter = 10
eps = 0.0001
loss = torch.nn.BCELoss()
savepath = Path("model_test.pch")
if False:
    print("Starting from latest checkpoint")
    with savepath.open("rb") as fp:
        state = torch.load(fp)
else:
    auto = AutoEncoder(next(iter(train))[0].shape[1], dim)
    auto = auto.to(device)
    optim = torch.optim.AdamW(auto.parameters(), lr=eps)
    state = State(auto, optim)

for epoch in range(state.epoch,max_iter):
    print("Epoch : ",epoch)
    for x,y in train:
        state.optim.zero_grad()
        x = x.to(device)
        xhat = state.model(x.float())
        ltrain = loss(xhat.float(),x.float())
        ltrain.backward()
        state.optim.step()
        state.iteration += 1
    # Testing current parameters
    for x,y in test:
        with torch.no_grad():
            x = x.to(device)
            xhat = state.model(x.float())
            ltest = loss(xhat.float(),x.float())

    if epoch == 1 or epoch == 25 or epoch == 49 :
        images = (x[0:3]).clone().detach().view(3,28,28).unsqueeze(1).repeat(1,3,1,1).float()
        images_pred = (xhat[0:3]).clone().detach().view(3,28,28).unsqueeze(1).repeat(1,3,1,1).float()
        # Permet de fabriquer une grille d'images
        images = make_grid(images)
        images_pred = make_grid(images_pred)
        # Affichage avec tensorboard
        writer.add_image(f'original/'+str(epoch), images, epoch)
        writer.add_image(f'pred/'+str(epoch), images_pred, epoch)

    writer.add_scalars('AutoEncoderTest/',{'train':ltrain,'test':ltest}, epoch)

    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save(state, fp)

