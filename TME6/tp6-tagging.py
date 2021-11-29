import itertools
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import datetime
from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
logging.basicConfig(level=logging.INFO)

ds = prepare_dataset('org.universaldependencies.french.gsd')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter("runs/runs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


class Vocabulary:
    """Permet de gérer un vocabulaire.
    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !
    Utilisation:
    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate(batch):
    """Collate using pad_sequence"""
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))


logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)
train_data = TaggingDataset(ds.train, words, tags, True)
dev_data = TaggingDataset(ds.validation, words, tags, True)
test_data = TaggingDataset(ds.test, words, tags, False)


logging.info("Vocabulary size: %d", len(words))


BATCH_SIZE = 64
EMB_SIZE = 100
H_SIZE = 10
NB_EPOCH = 5

train_loader = DataLoader(train_data, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate, batch_size=BATCH_SIZE)


#  TODO:  Implémentez le modèle et la boucle d'apprentissage



class TaggingModel(torch.nn.Module):

    def __init__(self, vocab_size, emb_size, h_size, tag_size):
        super(TaggingModel, self).__init__()
        self.h_size = h_size        

        self.embedding = torch.nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = torch.nn.LSTM(emb_size, h_size)
        self.linear = torch.nn.Linear(h_size, tag_size)

    def forward(self, x):
        emb = self.embedding(x)
        output, (hn, cn) = self.lstm(emb)
        return self.linear(output)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)
        
model = TaggingModel(words.__len__(), EMB_SIZE, H_SIZE, tags.__len__())
model = model.to(device)
model.apply(init_weights)
criterion = torch.nn.CrossEntropyLoss(ignore_index = 0)
optim = torch.optim.Adam(model.parameters(),lr=0.1)

def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / y[non_pad_elements].shape[0]



loss_train_value=[]
acc_train_value = []
loss_test_value=[]
acc_test_value = []
for i in range(50):
    loss_train = []
    acc_train = []
    for x,y in train_loader:
        optim.zero_grad()
        for t in np.random.randint(0,20,5) :
            for w in np.random.randint(0,len(x[:,t]),3) :     
                x[:,t][w]=1
        pred = model(x)
        predictions = pred.view(-1, pred.shape[-1])
        y = y.view(-1)
        loss = criterion(predictions, y)
        acc = categorical_accuracy(predictions, y, 0)
        acc_train.append(acc)
        loss_train.append(loss)
        loss.backward()
        optim.step()
       
        
        
    loss_test=[]
    acc_test =[]
    for x,y in test_loader:
        with torch.no_grad():
            pred = model(x)
            predictions = pred.view(-1, pred.shape[-1])
            y1 = y.view(-1)
            loss = criterion(predictions, y1)
            acc = categorical_accuracy(predictions, y1, 0)
            acc_test.append(acc)
            loss_test.append(loss)
    loss_train_value.append(torch.tensor(loss_train).mean())
    loss_test_value.append(torch.tensor(loss_test).mean())
    acc_train_value.append(torch.tensor(acc_train).mean())
    acc_test_value.append(torch.tensor(acc_test).mean())
    print("iteration {}, loss train : {}, acc train :{}, loss test : {}, acc test : {}".format(i,torch.tensor(loss_train).mean(),torch.tensor(acc_train).mean(),torch.tensor(loss_test).mean(),torch.tensor(acc_test).mean()))

fig, ax = plt.subplots(figsize=(10,5))
ax.grid()
ax.set(xlabel = "epochs", ylabel="Loss", title="loss en fonction de nombre epochs")
plt.plot(np.arange(len(loss_train_value)), loss_train_value, label="train")
plt.plot(np.arange(len(loss_test_value)), loss_test_value, label="test")
ax.legend()
plt.savefig(f"loss_tagging_")  

fig, ax = plt.subplots(figsize=(10,5))
ax.grid()
ax.set(xlabel = "epochs", ylabel="Accuracy", title="accuracy en fonction de nombre epochs")
plt.plot(np.arange(len(acc_train_value)), acc_train_value, label="train")
plt.plot(np.arange(len(acc_test_value)), acc_test_value, label="test")
ax.legend()
plt.savefig(f"acc_tagging_")  

