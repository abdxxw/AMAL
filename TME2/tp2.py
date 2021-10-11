import torch
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
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

# writer = SummaryWriter()

# data = datamaestro.prepare_dataset("edu.uci.boston")
# colnames, datax, datay = data.data()
# datax = torch.tensor(datax,dtype=torch.float)
# datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

"""
def RegLog(X,y,mini_batch,eps,max_iter) : 
    W = torch.randn(X.shape[1],1, requires_grad=True, dtype=torch.float)
    b = torch.randn(1,1, requires_grad=True, dtype=torch.float)
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

            writer.add_scalar('Loss/train', loss, max_iter)
    return W,b


def test(X_test,y_test,w,b):
    with torch.no_grad() : 
        y_test=X_test@w+b
        loss_test = torch.sum(torch.pow(y_test-y,2))

        writer.add_scalar('Loss/train', loss_test,100)
    return loss_test

x = torch.randn(50, 13)
y = torch.randn(50, 3)

w,b = RegLog(x,y,1,0.001,100)
print(w)

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

x = torch.randn(512,128)
y = torch.randn(512, 3)


import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module) : 
    def __init__(self) : 
        super(Model, self).__init__()
        self.fc1 = nn.Linear(128,64)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(64,3)

    def forward(self,x) : 
        return self.fc2(self.tanh(self.fc1(x)))

model = Model()
Loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
n_epochs = 100 
model.train()
for epoch in range(n_epochs) : 
    train_loss=0.0
    optimizer.zero_grad()
    output=model(x)
    loss = Loss(output,y)
    loss.backward()
    optimizer.step()
    writer.add_scalar('Loss/train_module', loss,100)
"""
##################Module with Sequential############################

data = datamaestro.prepare_dataset("housing.data")
colnames, datax, datay = data.data()
datax = torch.tensor(datax, dtype=torch.float)
datay = torch.tensor(datay, dtype=torch.float).reshape(-1, 1)

x = torch.randn(512, 128)
y = torch.randn(512, 3)

import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 3))

    def forward(self, x):
        return self.fc(x)


model = Model()
Loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
n_epochs = 100
model.train()
for epoch in range(n_epochs):
    train_loss = 0.0
    optimizer.zero_grad()
    output = model(x)
    loss = Loss(output, y)
    loss.backward()
    optimizer.step()
    print(loss)
    writer.add_scalar('Loss/train_module~_with_sequential', loss, 100)




