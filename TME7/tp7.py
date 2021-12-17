import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click

import datetime
from datamaestro import prepare_dataset

from torchvision import datasets, transforms
from torch.utils.data.sampler import RandomSampler



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self,in_size,out_size, batchnorm=False, layernorm=False, dropout=0.):
        super(Net, self).__init__()
        self.layer1 = []
        self.layer2 = []
        self.layer3 = []
        
        self.layer1.append(nn.Linear(in_size, 500))
        if batchnorm: self.layer1.append(nn.BatchNorm1d(500))
        if layernorm: self.layer1.append(nn.LayerNorm(500))
        self.layer1.append(nn.ReLU())
        if dropout: self.layer1.append(nn.Dropout(dropout))
        
        self.layer2.append(nn.Linear(500, 250))
        if batchnorm: self.layer2.append(nn.BatchNorm1d(250))
        if layernorm: self.layer2.append(nn.LayerNorm(250))
        self.layer2.append(nn.ReLU())
        if dropout: self.layer2.append(nn.Dropout(dropout))
        
        self.layer3.append(nn.Linear(250, 100))
        if batchnorm: self.layer3.append(nn.BatchNorm1d(100))
        if layernorm: self.layer3.append(nn.LayerNorm(100))
        self.layer3.append(nn.ReLU())
        if dropout: self.layer3.append(nn.Dropout(dropout))
        
        self.fc1 = nn.Sequential(*self.layer1)
        self.fc2 = nn.Sequential(*self.layer2)
        self.fc3 = nn.Sequential(*self.layer3)
        self.fc4 = nn.Linear(100, out_size)
        
    def forward(self, x):
        
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        x3 = self.fc3(x2)
        x4 = self.fc4(x3)
        return x1, x2, x3, x4

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
        
class DataMNIST(Dataset):

    def __init__(self, X, Y, transform=None):

        self.exemples = torch.from_numpy(X.reshape(X.shape[0],-1) / 255.0)
        self.orig = X / 255.0
        self.labels = torch.from_numpy(Y.copy()).float()
        self.len = self.exemples.shape[0]
        self.transform = transform

    def __getitem__(self, index):

            
        sample = self.exemples[index]

        if self.transform:
            sample = self.transform(self.orig[index])
        return sample, self.labels[index]

    def __len__(self):
        return self.len



def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var


def generate_hist_weights(writer, net, epoch):
    # for i, layer in enumerate(net):
    #     if isinstance(layer, nn.Linear):
    #         writer.add_histogram("layer"+str(i),layer.weight,epoch)
    l = [net.fc1,net.fc2,net.fc3]
    for i, layer in enumerate(l):
        writer.add_histogram("layer"+str(i),layer[0].weight,epoch)
    writer.add_histogram("layer"+str(4),net.fc4.weight,epoch)
            
def generate_hist_grad(writer, vars, epoch):
    for i, var in enumerate(vars):
        writer.add_histogram("input"+str(i),var.grad,epoch)
        

def generate_hist_entropy(writer, out, epoch):
    
    o = F.softmax(out, dim=1)
    o = o * torch.log(o)
    o = -o.sum(dim=1)
    writer.add_histogram('entropy', o, epoch)


def l2_reg(net,l2_lambda=0.01):
    l2_reg = torch.tensor(0.).to(device)
    for param in net.parameters():
        l2_reg += param.norm(p=2)
    return l2_lambda * l2_reg

def l1_reg(net,l1_lambda=0.01):
    l1_reg = torch.tensor(0.).to(device)
    for param in net.parameters():
        l1_reg += param.norm(p=1)
    return l1_lambda * l1_reg
    


TRAIN_RATIO = 0.05

loss = nn.CrossEntropyLoss()
max_iter = 1000
batch_size = 300
eps = 0.001

writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

ds = prepare_dataset("com.lecun.mnist")

train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

train_length = int(train_images.shape[0]*TRAIN_RATIO)

ind = torch.randperm(train_images.shape[0])
train_images, train_labels = train_images[ind][:train_length], train_labels[ind][:train_length]


train = DataLoader(DataMNIST(train_images,train_labels,transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               #transforms.RandomRotation(30),
                                               #transforms.RandomResizedCrop(28),
                                               #transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                               #AddGaussianNoise(0., 1.),
                                               transforms.Lambda(lambda x: torch.flatten(x)),
                                           ])),shuffle=True,batch_size=batch_size)

test = DataLoader(DataMNIST(test_images,test_labels),shuffle=True,batch_size=batch_size)

model = {   "net" : Net(28 * 28, 10,batchnorm=False,dropout=0),
            "l1_reg" : 0,
            "l2_reg" : 0
         }
model["net"].to(device)
optim = torch.optim.AdamW(model["net"].parameters(), lr=eps)

for e in range(max_iter):
    model["net"].train()
    ltrain = []
    for x,y in train:
        x = x.to(device)
        y = y.to(device)
        x.requires_grad = True
        optim.zero_grad()
        h1, h2, h3, yhat = model["net"](x.float())
        l = loss(yhat,y.long())
        
        l+= l1_reg(model["net"],model["l1_reg"]) + l2_reg(model["net"],model["l2_reg"])

        store_grad(x)
        h1.retain_grad()
        h2.retain_grad()
        h3.retain_grad()
        
        ltrain.append(l)
        l.backward()
        optim.step()
    ltrain = torch.tensor(ltrain).mean()
    if e % (max_iter/20) == 0: 
        generate_hist_weights(writer,model["net"],e)
        generate_hist_entropy(writer,yhat,e)
        generate_hist_grad(writer,[x,h1,h2,h3],e)
        print("done : epoch "+str(e))
        
    model["net"].eval()
    ltest = []
    for x,y in test:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            _, _, _, yhat = model["net"](x.float())
            ltest.append(loss(yhat,y.long()))

    ltest = torch.tensor(ltest).mean()
    print(ltrain,ltest)
    writer.add_scalars('Baseline/',{'train':ltrain,'test':ltest}, e)


