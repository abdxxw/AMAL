from utils import RNN, device,SampleMetroDataset
import torch
import torch.nn as nn

#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

train, test = torch.load("data/hzdataset.pch")
train = SampleMetroDataset(train)
test = SampleMetroDataset(test)
exp = sampleData[0]
print(exp)