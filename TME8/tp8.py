import logging

from torch.nn.modules.pooling import MaxPool1d
logging.basicConfig(level=logging.INFO)

import heapq
from pathlib import Path
import gzip
import math
from tqdm import tqdm

from typing import List

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm

#from tp8_preprocess import TextDataset

import torch.nn.functional as F

# Utiliser tp8_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration

# Taille du vocabulaire
vocab_size = 1000
MAINDIR = "/content"

# Chargement du tokenizer

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)

def loaddata(mode):
    with gzip.open(f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)


test = loaddata("test")
train = loaddata("train")
TRAIN_BATCHSIZE=500
TEST_BATCHSIZE=500


# --- Chargements des jeux de données train, validation et test

val_size = 1000
train_size = len(train) - val_size
train, val = torch.utils.data.random_split(train, [train_size, val_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
val_iter = torch.utils.data.DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_iter = torch.utils.data.DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)


        


class TextClassifier(nn.Module):

   def __init__(self,num_words,embedding_size,out_size,stride):
      super(TextClassifier, self).__init__()

      self.kernel_1 = 2
      self.kernel_2 = 3
      self.dim = 128
      self.num_words = num_words
      # Output size for each convolution
      self.out_size = out_size
      # Number of strides for each convolution
      self.stride = stride
      
      self.dropout = nn.Dropout(0.25)      
      self.embedding_size=embedding_size
    
      # Embedding layer definition
      self.embedding = nn.Embedding(self.num_words, self.embedding_size, padding_idx=0)
      
      # Convolution layers definition
      self.conv_1 = nn.Conv1d(self.embedding_size,self.dim, self.kernel_1, self.stride)
      self.conv_2 = nn.Conv1d(128,self.out_size, self.kernel_2, self.stride)

      
      # Max pooling layers definition
      self.pool_1 = nn.AdaptiveMaxPool1d(int(self.dim/2))
      self.pool_2 = nn.AdaptiveMaxPool1d(int(self.dim/4))

      
      
      self.fc = nn.Linear(96,self.out_size)
   def in_features_fc(self):
      '''Calculates the number of output features after Convolution + Max pooling
         
      Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
      Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1      
      '''
      # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
      out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
      out_conv_1 = math.floor(out_conv_1)
      out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
      out_pool_1 = math.floor(out_pool_1)
      
      # Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
      out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
      out_conv_2 = math.floor(out_conv_2)
      out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
      out_pool_2 = math.floor(out_pool_2)

      return out_pool_2
      
   def forward(self, x):

      # Sequence of tokes is filterd through an embedding layer
      bsize=x.size(0)
      x = self.embedding(x)
      x=x.permute(0,2,1)
      # Convolution layer 1 is applied
      x1 = self.conv_1(x)
      x1 = torch.relu(x1)
      x1 = self.pool_1(x1)
      # Convolution layer 2 is applied
      x2 = self.conv_2(x1)
      x2 = torch.relu((x2))
      x2 = self.pool_2(x2)    
      x2=x2.view(bsize,-1)
      

      out = self.fc(x2)
      # Dropout is applied		
      out = self.dropout(out)


      
      return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Modele(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout=0.5, pad_idx=0):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.emb=nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.features=nn.ModuleList([nn.Conv1d(embedding_dim, n_filters, 
                                             kernel_size=fs)for fs in filter_sizes])
        self.fc=nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.batch_norm = nn.BatchNorm1d(n_filters)
    def forward(self,x):
        x_emb=self.emb(x).permute(0,2,1)
        f=[F.relu(conv(x_emb)) for conv in self.features]
        #f = [self.batch_norm(f1) for f1 in f]
        fpooled=[F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in f]
        outf = self.dropout(torch.cat(fpooled, dim = 1))

        return self.fc(outf)

"""     
max_iter = 100
eps = 0.001
loss = torch.nn.CrossEntropyLoss()

embedding_dim = 300
n_filters = 100
filter_sizes = [3,4]
output_dim = 3
dropout_rate = 0.25





model = TextClassifier(vocab_size,embedding_dim,output_dim,stride=1)

for p in model.parameters():
  if p.dim() > 1:
    nn.init.xavier_uniform_(p)
model = model.to(device)

optim = torch.optim.AdamW(model.parameters(), lr=eps)

acc_list=[]
loss_list=[]
loss_test_list = []
acc_test_list = []
acc_val_list = []
loss_val_list = []
for i in range(100): 
    model.train()
    loss_train_value=[]
    acc=[]
    for x,y in train_iter : 
        #Training Step
        x=x.to(device)
        y=y.to(device)
        optim.zero_grad()
        y_pred = model(x.long())
        y_pred = y_pred.to(device)
        loss_train = loss(y_pred,y.long()) 
        loss_train = loss_train
        loss_train.backward()
        acc.append(sum((y_pred.argmax(1) == y.reshape(-1))).item() / y.shape[0])
        #print("loss",loss_train)
        optim.step()
        loss_train_value.append(loss_train)
        #Validation step
        """
        with torh.no_grad() : 
          for x_val,y_val in val_iter : 
            x_val=x_val.to(device)
            y_val=y_val.to(device)
            y_pred = model(x_val.long())
            loss_test = loss(y_pred,y.long())
            loss_val_list.append(loss_test)
            acc_val_test.append(sum((y_pred.argmax(1) == y.reshape(-1))).item() / y.shape[0])
        """
    model.eval()
    acc_list.append(torch.tensor(acc).mean())
    print("Train",torch.tensor(acc).mean())
    loss_list.append(torch.tensor(loss_train_value).mean())
    loss_value=[]
    acc = []
    for x,y in test_iter :
        with torch.no_grad() : 
            x=x.to(device)
            y=y.to(device)
            y_pred = model(x.long())
            y_pred = y_pred.to(device)
            loss_test = loss(y_pred,y.long())
            loss_value.append(loss_test)
            acc.append(sum((y_pred.argmax(1) == y.reshape(-1))).item() / y.shape[0])
    print("Test",torch.tensor(acc).mean())
    acc_test_list.append(acc)
    loss_test_list.append(torch.tensor(loss_value).mean())