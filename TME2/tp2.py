import torch
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

writer = SummaryWriter()
"""

a = torch.rand((1,10),requires_grad=True)
b = torch.rand((1,10),requires_grad=True)
c = a.mm(b.t())
d = 2 * c
c.retain_grad() # on veut conserver le gradient par rapport à c
d.backward() ## calcul du gradient et retropropagation
##jusqu’aux feuilles du graphe de calcul
print(d.grad) #Rien : le gradient par rapport à d n’est pas conservé
print(c.grad) # Celui-ci est conservé
print(a.grad) ## gradient de d par rapport à a qui est une feuille
print(b.grad) ## gradient de d par rapport à b qui est une feuille
d = 2 * a.mm(b.t())
d.backward()
print(a.grad) ## 2 fois celui d’avant, le gradient est additioné
a.grad.data.zero_() ## reinitialisation du gradient pour a
d = 2 * a.mm(b.t())
d.backward()
print(a.grad) ## Cette fois, c’est ok
with torch.no_grad():
    c = a.mm(b.t()) ## Le calcul est effectué sans garder le graphe de calcul
#c.backward() ## Erreur
"""

writer = SummaryWriter()
random.seed(42)
data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(datax, datay, test_size=0.4, random_state=42)

dataX = (Xtrain - Xtrain.min(axis=0)) / (Xtrain.max(axis=0) - Xtrain.min(axis=0))
dataX_train = torch.tensor(dataX, dtype=torch.float)
dataY_train = torch.tensor(Ytrain, dtype=torch.float).reshape(-1, 1)
dataX_test = (Xtest - Xtest.min(axis=0)) / (Xtest.max(axis=0) - Xtest.min(axis=0))
dataX_test = torch.tensor(dataX_test, dtype=torch.float)
dataY_test = torch.tensor(Ytest, dtype=torch.float).reshape(-1, 1)

print(dataX_train.shape)
print(dataY_train.shape)
print(dataX_test.shape)
print(dataY_test.shape)

"""

def RegLog(X,y,mini_batch,eps,max_iter) : 
    acc=[]
    i=0
    W = torch.randn(X.shape[1],y.shape[1], requires_grad=True, dtype=torch.float)
    b = torch.randn(y.shape[1],1, requires_grad=True, dtype=torch.float)
    for i in range(max_iter) : 
        ind = torch.randperm(X.shape[0])
        for j in range(0,X.shape[0],mini_batch) : 
            indices = ind[j:j+mini_batch]
            X_batch,y_batch = X[indices],y[indices]
            yhat = X_batch @ W + b
            loss = torch.sum(torch.pow(yhat-y_batch,2))
            loss.backward()

            W.data = W-eps*W.grad
            b.data = b - eps*b.grad
            W.grad.data.zero_()
            b.grad.data.zero_() 

            writer.add_scalar('Loss/train', loss, i)
    return W,b


def test(X,y,w,b,max_iter,mini_batch):
    for i in range(max_iter) : 
        ind = torch.randperm(X.shape[0])
        for j in range(0,X.shape[0],mini_batch) : 
            indices = ind[j:j+mini_batch]
            X_batch,y_batch = X[indices],y[indices]
            with torch.no_grad() : 
                y_hat=X_batch@w+b
                loss_test = torch.sum(torch.pow(y_batch-y_hat,2))
                writer.add_scalar('Loss/test', loss_test,i)
    return loss_test



w,b = RegLog(dataX_train,dataY_train,1,0.001,100)
test(dataX_test,dataY_test,w,b,100,1)

"""

####################Optimiseur#########################
"""
w = torch.nn.Parameter(torch.randn(1,10))
b = torch.nn.Parameter(torch.randn(1))
EPS=0.01
optim = torch.optim.SGD(params=[w,b],lr=EPS) ## on optimise selon w et b, lr : pas de gradient
optim.zero_grad()
NB_EPOCH = 100
# Reinitialisation du gradient
for i in range(NB_EPOCH):
    loss = MSE(f(x,w,b),y) #Calcul du cout
    loss.backward() # Retropropagation
    if i % 100 = 0:
        optim.step() # Mise-à-jour des paramètres w et b
        optim.zero_grad() # Reinitialisation du gradient
"""
#################### Module ##############################

"""
dim=dataX_train.shape[1]


import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module) : 
    def __init__(self,dim) : 
        super(Model, self).__init__()
        self.fc1 = nn.Linear(dim,64)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(64,1)

    def forward(self,x) : 
        return self.fc2(self.tanh(self.fc1(x)))

model = Model(dim)
Loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
n_epochs = 100 
batch = 10
model.train()
for epoch in range(n_epochs) : 
    ind = torch.randperm(dataX_train.shape[0])
    for j in range(0,dataX_train.shape[0],10): 
        indices = ind[j:j+batch]
        X_batch,y_batch = dataX_train[indices],dataY_train[indices]

        train_loss=0.0
        optimizer.zero_grad()
        output=model(X_batch)
        loss = Loss(output,y_batch)
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train_module', loss,epoch)

    model.eval()
    ind = torch.randperm(dataX_test.shape[0])
    for j in range(0,dataX_test.shape[0],10): 
        indices = ind[j:j+batch]
        X_batch,y_batch = dataX_test[indices],dataY_test[indices]   

        with torch.no_grad():            
            predictions = model(X_batch).squeeze()
            loss = Loss(predictions,y_batch)
            writer.add_scalar('Loss/test_module', loss,epoch)

"""
##################Module with Sequential############################


dim = dataX_train.shape[1]

import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        l1 = nn.Linear(dim, 64)
        l2 = nn.Linear(64, 1)
        torch.nn.init.xavier_uniform(l1.weight)
        torch.nn.init.xavier_uniform(l2.weight)
        self.fc = nn.Sequential(l1, nn.Tanh(), l2)

    def forward(self, x):
        return self.fc(x)


model = Model(dim)
Loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
n_epochs = 100
batch = 100
for epoch in range(n_epochs):
    ind = torch.randperm(dataX_train.shape[0])
    for j in range(0, dataX_train.shape[0], batch):
        indices = ind[j:j + batch]
        X_batch, y_batch = dataX_train[indices], dataY_train[indices]

        train_loss = 0.0
        optimizer.zero_grad()
        output = model(X_batch)
        loss = Loss(output, y_batch)
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train_module', loss, epoch)

    model.eval()
    ind = torch.randperm(dataX_test.shape[0])
    for j in range(0, dataX_test.shape[0], batch):
        indices = ind[j:j + batch]
        X_batch, y_batch = dataX_test[indices], dataY_test[indices]

        with torch.no_grad():
            predictions = model(X_batch).squeeze()
            loss = Loss(predictions, y_batch)
            writer.add_scalar('Loss/test_module', loss, epoch)


