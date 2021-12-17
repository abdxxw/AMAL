import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,random_split
from pathlib import Path
from datamaestro import prepare_dataset
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from torchvision import transforms

BATCH_SIZE = 300
TRAIN_RATIO = 0.05
LOG_PATH = "/tmp/runs/lightning_logs"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Lit2Layer(pl.LightningModule):

    def __init__(self,in_size,out_size,learning_rate=1e-3, l1=0, l2=0, batchnorm=False, layernorm=False, dropout=0.):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.name = "mnist-lightning"
        self.l1 = l1
        self.l2 = l2
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


    def l2_reg(self):
        l2_reg = torch.tensor(0.).to(device)
        for param in self.parameters():
            l2_reg += param.norm(p=2)
        return self.l2 * l2_reg

    def l1_reg(self):
        l1_reg = torch.tensor(0.).to(device)
        for param in self.parameters():
            l1_reg += param.norm(p=1)
        return self.l1 * l1_reg
        
    def configure_optimizers(self):
        """ Définit l'optimiseur """
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.learning_rate)
        return optimizer

    def training_step(self,batch,batch_idx):
        """ une étape d'apprentissage
        doit retourner soit un scalaire (la loss),
        soit un dictionnaire qui contient au moins la clé 'loss'"""
        x, y = batch

        h1, h2, h3, yhat= self(x.float()) ## equivalent à self.model(x)
        
  
        loss = self.loss(yhat,y.long())
        loss+= self.l1_reg() + self.l2_reg()

        
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x),"x":x,"h1":h1,"h2":h2,"h3":h3,"yhat":yhat}
        self.log("train_accuracy",acc/len(x),on_step=False,on_epoch=True)
        return logs

    def validation_step(self,batch,batch_idx):
        """ une étape de validation
        doit retourner un dictionnaire"""
        x, y = batch
        h1, h2, h3, yhat = self(x.float())
        loss = self.loss(yhat,y.long())
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        self.log("val_accuracy", acc/len(x),on_step=False,on_epoch=True)
        return logs

    def test_step(self,batch,batch_idx):
        """ une étape de test """
        x, y = batch
        h1, h2, h3, yhat = self(x.float())
        loss = self.loss(yhat,y.long())
        acc = (yhat.argmax(1)==y).sum()
        logs = {"loss":loss,"accuracy":acc,"nb":len(x)}
        self.log("test_accuracy",acc/len(x),on_step=False,on_epoch=True)
        return logs

    def training_epoch_end(self,outputs):
        """ hook optionel, si on a besoin de faire quelque chose apres une époque d'apprentissage.
        Par exemple ici calculer des valeurs à logger"""
        
        e = self.current_epoch
        
        if e % (50) == 0: 

          o = F.softmax(outputs[-1]["yhat"], dim=1)
          o = o * torch.log(o)
          o = -o.sum(dim=1)
          self.logger.experiment.add_histogram('entropy', o, e)

          l = [self.fc1,self.fc2,self.fc3]
          for i, layer in enumerate(l):
              self.logger.experiment.add_histogram("layer"+str(i),layer[0].weight,e)
          self.logger.experiment.add_histogram("layer"+str(4),self.fc4.weight,e)

        total_acc = sum([o['accuracy'] for o in outputs])
        total_nb = sum([o['nb'] for o in outputs])
        total_loss = sum([o['loss'] for o in outputs])/len(outputs)
        total_acc = total_acc/total_nb
        self.log_dict({f"loss/train":total_loss,f"acc/train":total_acc})
        # Le logger de tensorboard est accessible directement avec self.logger.experiment.add_XXX

    def validation_epoch_end(self, outputs):
        """ hook optionel, si on a besoin de faire quelque chose apres une époque de validation."""
        total_acc = sum([o['accuracy'] for o in outputs])
        total_nb = sum([o['nb'] for o in outputs])
        total_loss = sum([o['loss'] for o in outputs])/len(outputs)
        total_acc = total_acc/total_nb
        self.log_dict({f"loss/val":total_loss,f"acc/val":total_acc})

    def test_epoch_end(self, outputs):
        pass



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


class LitMnistData(pl.LightningDataModule):

    def __init__(self,batch_size=BATCH_SIZE,train_ratio=TRAIN_RATIO):
        super().__init__()
        self.dim_in = 784
        self.dim_out = 10
        self.batch_size = batch_size
        self.train_ratio = train_ratio

    def prepare_data(self):
        ### Do not use "self" here.
        prepare_dataset("com.lecun.mnist")

    def setup(self,stage=None):
        ds = prepare_dataset("com.lecun.mnist")
                
        images, labels = ds.train.images.data(), ds.train.labels.data()
        test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

        if stage =="fit" or stage is None:
            # Si on est en phase d'apprentissage
            shape = ds.train.images.data().shape
            self.dim_in = shape[1]*shape[2]
            self.dim_out = len(set(ds.train.labels.data()))
                        
            train_length = int(images.shape[0]*TRAIN_RATIO)

            ind = torch.randperm(images.shape[0])
            train_images, train_labels = images[ind][:train_length],labels[ind][:train_length]
            val_images, val_labels = images[ind][:-train_length], labels[ind][:-train_length]

            self.mnist_train, self.mnist_val, = DataMNIST(train_images,train_labels,transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               #transforms.RandomRotation(30),
                                               #transforms.RandomResizedCrop(28),
                                               #transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                               #AddGaussianNoise(0., 1.),
                                               transforms.Lambda(lambda x: torch.flatten(x)),
                                           ])), DataMNIST(val_images,val_labels)
        if stage == "test" or stage is None:
            # en phase de test
            self.mnist_test= DataMNIST(test_images,test_labels)

    def train_dataloader(self):
        return DataLoader(self.mnist_train,batch_size=self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.mnist_val,batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(self.mnist_test,batch_size=self.batch_size)



data = LitMnistData()

data.prepare_data()
data.setup()

model = Lit2Layer(data.dim_in,data.dim_out,learning_rate=1e-3,dropout=.25,l2=0.01,batchnorm=True)

logger = TensorBoardLogger(save_dir=LOG_PATH,name=model.name,version=time.asctime(),default_hp_metric=False)

trainer = pl.Trainer(gpus=1,default_root_dir=LOG_PATH,logger=logger,max_epochs=1000)
trainer.fit(model,data)
trainer.test(model,data)