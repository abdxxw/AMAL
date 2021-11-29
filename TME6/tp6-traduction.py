import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import torch.distributions.categorical as categorical
from torch.utils.tensorboard import SummaryWriter
import time
import re
import datetime
from torch.utils.tensorboard import SummaryWriter
import random

logging.basicConfig(level=logging.INFO)

writer = SummaryWriter("runs/runs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

FILE = "eng-fra.txt"
MAX_LEN = 10
BATCH_SIZE = 100
H_SIZE = 32
EMB_SIZE = 32
teacher_forcing_ratio = 0.5
writer = SummaryWriter()

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


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
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
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

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=MAX_LEN):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, emb_size, h_size):
        super(Encoder, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.gru = torch.nn.GRU(emb_size, h_size)
  

    def forward(self, x):
        x_emb = self.embedding(x)
        _, h = self.gru(x_emb)
        return h


class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, emb_size, h_size):
        super(Decoder, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.gru = torch.nn.GRU(emb_size, h_size)
        self.linear = torch.nn.Linear(h_size, vocab_size)
        self.relu = torch.nn.ReLU()


    def forward(self, x, hidden):
        seq_len, batch_size = x.shape
        vec = torch.ones(batch_size)*2
        vec = vec.long()
        vec=vec.unsqueeze(0)
        deb_emb = self.relu(self.embedding(vec))
        _, hidden_state = self.gru(deb_emb, hidden)
        output = self.linear(hidden_state)
        all_output = [output]
        
        for i in range(seq_len-1):
            x_emb = self.relu(self.embedding(x[i].unsqueeze(0)))
            _, hidden_state = self.gru(x_emb, hidden_state)
            output = self.linear(hidden_state)
            all_output.append(output)
        
        return torch.stack(all_output, 2)
        
    def generate(self, h, lenseq=None):
        batch_size = h.shape[1]
        vec = torch.ones(batch_size)*2
        vec = vec.long()
        vec=vec.unsqueeze(0)
        all_output = []
        while len(all_output) < lenseq:
            deb_emb = self.relu(self.embedding(vec))
            _, h = self.gru(deb_emb, h)
            output = self.linear(h)
            all_output.append(output)
            vec = torch.argmax(output, 2)
        return torch.stack(all_output, 2)       



model_encoder = Encoder(vocEng.__len__(), EMB_SIZE, H_SIZE)
model_decoder = Decoder(vocFra.__len__(), EMB_SIZE, H_SIZE)

criterion = torch.nn.CrossEntropyLoss()

optim_encoder = torch.optim.Adam(model_encoder.parameters(), lr=0.01)
optim_decoder = torch.optim.Adam(model_decoder.parameters(), lr=0.01)




for i in range(50):
    losstrain = []
    losstest = []
    for x in train_loader:
        optim_encoder.zero_grad()
        optim_decoder.zero_grad()
        hn = model_encoder(x[0]) 
        if random.random() < teacher_forcing_ratio :
            output = model_decoder(x[2], hidden=hn)
        else:
            output = model_decoder.generate(hn, lenseq=len(x[2]))
        
        loss = criterion(torch.transpose(output.squeeze(0), 1, 2), x[2].T)
        loss.backward()
        optim_encoder.step()
        optim_decoder.step()
        losstrain.append(loss) 
    for x in test_loader:
        with torch.no_grad():
            hn = model_encoder(x[0])
            output = model_decoder.generate(hn, lenseq=len(x[2]))
            loss = criterion(torch.transpose(output.squeeze(0), 1, 2), x[2].T)
            losstest.append(loss)       
            if i == 98 or i==99 or i==97 or i==96:
                pred = torch.argmax(torch.squeeze(output, 0), 2)
                print("sequence prédit : ", vocFra.getwords(pred.T[:,-1]))
                print("séquence vrai : ", vocFra.getwords(x[2][:,-1]))
    
    print("iteration {}, loss test : {}".format(i,torch.tensor(losstest).mean()))
    writer.add_scalars('Texttranslation', {'train': torch.tensor(losstrain).mean(),'test': torch.tensor(losstest).mean()}, i)