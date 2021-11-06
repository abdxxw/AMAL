
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *
import datetime
from pathlib import Path
#  TODO:

class State(object) :
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0, 0

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    unrduced_loss = nn.functional.cross_entropy(output,target,reduction='none')
    mask = target != padcar
    loss_masked = unrduced_loss.where(mask, torch.tensor(0.0).to(device))
    reduced_loss = loss_masked.sum()/ mask.sum()
    return reduced_loss

class RNN(nn.Module):
    #  TODO:  Recopier l'implémentation du RNN (TP 4)
    def __init__(self, dim, latent, output):

        super(RNN, self).__init__()
        self.dim = dim
        self.latent = latent
        self.output = output

        self.in_layer = nn.Linear((self.dim + self.latent), self.latent)
        self.out_layer = nn.Linear(self.latent, self.output)
        self.tanh = nn.Tanh()

    def one_step(self, x, h=None):

        if h is None:
            h = torch.zeros(x.shape[0],self.latent).to(device)
        temp = torch.cat((x, h), 1)
        return self.tanh(self.in_layer(temp))

    def forward(self, x, h=None):

        if h is None:
            h = torch.zeros(x.shape[0], self.latent).to(device)
        h_history = [h]
        for i in range(x.shape[1]):
            h_history.append(self.one_step(x[:, i, :], h_history[-1]))

        return h_history[1:]

    def decode(self, h):
        return self.out_layer(h)

class LSTM(nn.Module):
    #  TODO:  Implémenter un LSTM
    def __init__(self, dim, latent, output):

        super(LSTM, self).__init__()
        self.dim = dim
        self.latent = latent
        self.output = output

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        self.i = nn.Linear((self.dim + self.latent), self.latent)
        self.f = nn.Linear((self.dim + self.latent), self.latent)
        self.o = nn.Linear((self.dim + self.latent), self.latent)
        self.in_layer = nn.Linear((self.dim + self.latent), self.latent)
        self.out_layer = nn.Linear(self.latent, self.output)

    def one_step(self, x, h=None, C= None):

        if h is None:
            h = torch.zeros(x.shape[0],self.latent).to(device)
        if C is None:
            C = torch.zeros(x.shape[0],self.latent).to(device)
        temp = torch.cat((x, h), 1)
        keep = self.sig(self.i(temp))
        forget = self.sig(self.f(temp))
        out = self.sig(self.o(temp))
        Ct = forget * C + keep * self.tanh(self.in_layer(temp))
        return out * self.tanh(Ct), Ct

    def forward(self, x, h=None, C=None):

        if h is None:
            h = torch.zeros(x.shape[0], self.latent).to(device)
        if C is None:
            C = torch.zeros(x.shape[0],self.latent).to(device)
        h_history = [h]
        for i in range(x.shape[1]):
            h, C = self.one_step(x[:, i, :], h_history[-1],C)
            h_history.append(h)

        return h_history[1:], C

    def decode(self, h):
        return self.out_layer(h)



class GRU(nn.Module):
    #  TODO:  Implémenter un GRU
    def __init__(self, dim, latent, output):

        super(GRU, self).__init__()
        self.dim = dim
        self.latent = latent
        self.output = output

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        self.z = nn.Linear((self.dim + self.latent), self.latent)
        self.r = nn.Linear((self.dim + self.latent), self.latent)
        self.in_layer = nn.Linear((self.dim + self.latent), self.latent)
        self.out_layer = nn.Linear(self.latent, self.output)

    def one_step(self, x, h=None):

        if h is None:
            h = torch.zeros(x.shape[0],self.latent).to(device)
        temp = torch.cat((x, h), 1)
        zt = self.sig(self.z(temp))
        rt = self.sig(self.r(temp))
        return (1-zt) * h + zt * self.tanh(self.in_layer(torch.cat((x, rt * h), 1)))

    def forward(self, x, h=None):

        if h is None:
            h = torch.zeros(x.shape[0], self.latent).to(device)
        h_history = [h]
        for i in range(x.shape[1]):
            h_history.append(self.one_step(x[:, i, :], h_history[-1]))

        return h_history[1:]

    def decode(self, h):
        return self.out_layer(h)



#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot

writer = SummaryWriter("runs/runs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

PATH = "data/"
BATCH_SIZE = 32
max_len = 30

with open(PATH + "trump_full_speech.txt", "r") as f:
    txt = f.read()

ds = TextDataset(txt, maxlen=max_len)
data = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=pad_collate_fn, shuffle=True)

max_iter = 1001
lr = 0.001
DIM_HIDDEN = 100
dimbed = 80
soft = nn.LogSoftmax(dim=1)


rnn = RNN(dimbed, DIM_HIDDEN, len(lettre2id)).to(device)
emb = nn.Embedding(len(id2lettre), dimbed, padding_idx=0).to(device)

def train_text_generation(path,rnn, emb, data, lr, max_iter):

    savepath = Path(path)
    if savepath.is_file():
        print("Starting from latest checkpoint")
        with savepath.open("rb") as fp:
            state = torch.load(fp)
    else:
        optim = torch.optim.AdamW(rnn.parameters(), lr)
        state = State(rnn, optim)

    loss = maskedCrossEntropy

    print("Training for {} epochs".format(max_iter))

    for iter in range(state.epoch,max_iter):
        losstrain = []
        acctrain = []

        for x in data:
            x = x.to(device)
            h = None
            embed = emb(x)
            # yhat = state.model(embed[:len(x) - 1])
            # decoded = torch.stack([state.model.decode(i) for i in yhat], 2)
            # y = x[1:]
            # l = loss(decoded, y, padcar=PAD_IX)
            l = 0
            for t in range(len(x) - 1):
                h = state.model.one_step(embed[t], h)
                yhat = soft(rnn.decode(h))
                l+= loss(yhat,x[t+1],PAD_IX)
            l = l/(len(x) - 1)
            l.backward()
            state.optim.step()
            state.optim.zero_grad()
            state.iteration += 1

            losstrain.append(l)

        if iter % 10 == 0:
            print("iteration {}, loss train : {}".format(iter,
                                                         torch.tensor(losstrain).mean()))

        writer.add_scalars('TextGeneration/', {'train': torch.tensor(losstrain).mean()}, iter)

    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save(state, fp)