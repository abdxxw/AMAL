import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np

from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text(encoding='UTF-8') if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text(encoding='UTF-8')), self.filelabels[ix]



def collate_batch(batch):

    data = [torch.tensor(item[0]) for item in batch]

    labels = [item[1] for item in batch]
    new_data = pad_sequence(data,padding_value=400000)
    new_data = new_data.transpose(0,1)
    return torch.tensor(new_data),torch.tensor(labels)


def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    words.append("__OOV__")
    PAD = len(words)
    words.append("__PAD__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)



word2id,embeddings,train,test = get_imdb_data()



BATCH_SIZE = 64

train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,collate_fn=collate_batch)
test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True,collate_fn=collate_batch)


class BaseLine(nn.Module):
    def __init__(self, taille_embedding, embedding):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding)
        self.fc = nn.Linear(taille_embedding,2)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x,dim=1)
        print(x.shape)
        x = self.fc(x)
        return x
    
class AttentionSimple(nn.Module):
    def __init__(self, taille_embedding, embedding):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding)
        self.taille_embedding=taille_embedding
        self.soft_max = nn.Softmax()
        self.q = nn.Parameter(torch.ones(taille_embedding))
        torch.nn.init.uniform_(self.q, a=0.0, b=1.0)
        self.fc = nn.Linear(taille_embedding,2)

    def forward(self, x):
        x = self.embedding(x) 
        q_exp = self.q.expand((x.shape[0],x.shape[1],self.taille_embedding)) 
        alpha=torch.sum(torch.mul(q_exp,x),dim=-1) 
        alpha = torch.nn.functional.softmax(alpha,dim=-1) 
        x = torch.mul(x,alpha.unsqueeze(2))
        x = torch.sum(x,dim=1)
        x = self.fc(x)
        return x,alpha
    
class AttentionQustionValue(nn.Module):
    def __init__(self, taille_embedding, embedding):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding)
        self.taille_embedding=taille_embedding
        self.q = nn.Linear(taille_embedding,taille_embedding)
        self.fc = nn.Linear(taille_embedding,2)
        self.v = nn.Linear(taille_embedding,taille_embedding)

    def forward(self, x):
        
        x = self.embedding(x) 
        t_hat = torch.mean(x,dim=1)
        x1 = self.q(t_hat)    
        q_exp = x1.expand((x.shape[1],x.shape[0],self.taille_embedding)) 
        q_exp = q_exp.transpose(0,1)
        alpha=torch.sum(torch.mul(q_exp,x),dim=-1) 
        alpha = torch.nn.functional.softmax(alpha,dim=-1)    
        x = torch.mul(self.v(x),alpha.unsqueeze(2))
        x = torch.sum(x,dim=1)
        x = self.fc(x)
        return x,alpha
    
    
model = BaseLine(50,torch.FloatTensor(embeddings))


learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()    # Softmax is internally computed.
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
training_epochs = 100

for i in range(training_epochs) : 
    model.train()
    for data,label in train_loader : 
        y_pred = model(data)
        loss = criterion(y_pred,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(sum((y_pred.argmax(1) == label.reshape(-1))).item() / label.shape[0])
   
    model.eval()
    for data,label in test_loader : 
        with torch.no_grad():
            y_pred = model(data)
            loss = criterion(y_pred,torch.tensor(label))
            print(sum((y_pred.argmax(1) == label.reshape(-1))).item() / label.shape[0])
            
            
        
        

      



