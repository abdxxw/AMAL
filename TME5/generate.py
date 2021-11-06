from textloader import  string2code, id2lettre
import math
import torch

from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  TODO:  Ce fichier contient les différentes fonction de génération

soft = nn.LogSoftmax(dim=1)

def generate(rnn, emb, decoder, eos, start="", maxlen=200):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """

    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
    if start == "":
        print("\ngenerating from nothing, taking most probable char : ")

        h = None
        generated = [torch.tensor(torch.randint(len(id2lettre),(1,))).to(device)]
        i=0
        while generated[-1] != eos and i < maxlen:
            h = rnn.one_step(emb(generated[-1]), h)
            generated.append(decoder(h).argmax(1))
            i+=1
        generated = torch.stack(generated[1:])
        print("".join([id2lettre[int(i)] for i in generated.squeeze()]))


        print("\ngenerating from nothing, taking random char : ")

        h = None
        generated = [torch.tensor([0]).to(device)]
        i=0
        while generated[-1] != eos and i < maxlen:
            h = rnn.one_step(emb(generated[-1]).to(device), h)
            prob = Categorical(logits=decoder(h))
            generated.append(prob.sample())
            i+=1
        generated = torch.stack(generated[1:])
        print("".join([id2lettre[int(i)] for i in generated.squeeze()]))

    else :
        print("\ngenerating from start : "+start+" , taking most probable char : ")
        h = None
        yhat = rnn(emb(string2code(x).view(1, -1)).to(device))
        generated = [decoder(yhat[-1]).argmax(1)]
        i=0
        while generated[-1] != eos and i < maxlen:
            h = rnn.one_step(emb(generated[-1]).float(), h)
            generated.append(decoder(h).argmax(1))
            i+=1
        generated = torch.stack(generated[1:])
        print(x + "".join([id2lettre[int(i)] for i in generated.squeeze()]))

def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez le beam Search
    if start == "":
        print("\ngenerating from nothing, using beam search : ")
        rnd = torch.randint(len(id2lettre),(1,))
        print(id2lettre[int(rnd[0])])
        tmp = torch.tensor(rnd).to(device)
        h = rnn.one_step(emb(tmp).float())
    else :

        print("\ngenerating from start : "+start+" , taking most probable char : ")
        h = rnn(emb(string2code(start).view(1, -1).to(device)).float())[-1]

    candidats = soft(decoder(h)).squeeze()
    candidats = [([i], s) for i, s in enumerate(candidats.tolist())]
    candidats = sorted(candidats, key=lambda x: -x[1])[:k]

    for i in range(1, maxlen):
        currenttop = []
        for x in candidats:
            h = rnn(emb(torch.tensor(x[0]).view(1, -1).to(device)))[-1]
            topk = soft(decoder(h)).squeeze(0)
            topk = [([i], s) for i, s in enumerate(topk.tolist())]
            currenttop += [(x[0] + j, x[1] + s) for j, s in topk]

        candidats = sorted(currenttop, key=lambda x: -x[1])[:k]
    best = candidats[0][0]

    if eos in best:
        end = best.index(eos)
        if end > 0:
            best = best[:best.index(eos)]
    print("".join([id2lettre[int(i)] for i in best]))



# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
        prob = soft(decoder(h)).squeeze(0)
        sorted, indices = torch.sort(prob, descending=True)
        top_indices = indices[sorted.cumsum(0) < 0.4]
        top_indices = torch.cat((top_indices,indices[len(top_indices)].view(1)))
        top = sorted[top_indices]
        not_wanted = prob.detach().clone()
        not_wanted[top_indices] = 0


        return (prob - not_wanted)/ top.sum()

    return compute
