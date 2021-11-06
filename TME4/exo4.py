import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader
import datetime

from utils import RNN, device

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]



#  TODO: 

writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


PATH = "data/"
BATCH_SIZE = 32
max_sent = 100
max_len = 20

with open(PATH+"trump_full_speech.txt","r") as f:
  txt = f.read()

ds = TrumpDataset(txt,maxlen=max_len)
data = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

max_iter = 1001
lr = 0.001
DIM_HIDDEN = 100

rnn = RNN(len(lettre2id), DIM_HIDDEN, len(lettre2id)).to(device)
layer = nn.Linear(len(lettre2id), len(lettre2id)).to(device)

def train_text_generation(rnn,layer,data,lr,max_iter):


    loss = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(rnn.parameters(), lr)

    print("Training for {} epochs".format(max_iter))

    for iter in range(max_iter):
      losstrain = []
      acctrain = []

      for x,y in data:
        x = x.to(device)
        y = y.to(device)
        oh = layer(nn.functional.one_hot(x,num_classes=len(lettre2id)).float().to(device))
        yhat = rnn(oh)
        decoded = torch.stack([rnn.decode(i) for i in yhat],2)
        l = loss(decoded,y)
        l.backward()
        optim.step()
        optim.zero_grad()

        losstrain.append(l)

      if iter % 10 == 0:
        print("iteration {}, loss train : {}".format(iter,
                                  torch.tensor(losstrain).mean()))

      writer.add_scalars('TextGeneration/',{'train':torch.tensor(losstrain).mean()}, iter)

def generate_from_empty(rnn,layer,size,max_len):

    for _ in range(size):
        h = None
        generated = [torch.tensor(torch.randint(len(lettre2id), (1,))).to(device)]
        for i in range(max_len):
            h = rnn.one_step(layer(nn.functional.one_hot(generated[-1], num_classes=len(lettre2id)).float()), h)
            generated.append(rnn.decode(h).argmax(1))
        generated = torch.stack(generated[1:])
        print("".join([id2lettre[int(i)] for i in generated.squeeze()]))


def generate_from_begining(rnn,layer,size,max_len):

    begin = [ds.phrases[int(torch.randint(len(ds.phrases), (1,))[0])][:5] for _ in range(size)]

    for x in begin:
        h = None
        oh = layer(nn.functional.one_hot(string2code(x).view(1, -1), num_classes=len(lettre2id)).float().to(device))
        yhat = rnn(oh)
        generated = [rnn.decode(yhat[-1]).argmax(1)]
        for i in range(max_len):
            h = rnn.one_step(layer(nn.functional.one_hot(generated[-1], num_classes=len(lettre2id)).float()), h)
            generated.append(rnn.decode(h).argmax(1))
        generated = torch.stack(generated[1:])
        print(x + "".join([id2lettre[int(i)] for i in generated.squeeze()]))