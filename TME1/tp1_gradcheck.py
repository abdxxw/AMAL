import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
mes_test=torch.autograd.gradcheck(mse, (yhat, y))
print(mes_test)
#  TODO:  Test du gradient de Linear (sur le même modèle que MSE)


X = torch.randn(8,5, requires_grad=True, dtype=torch.float64)
W = torch.randn(5,2, requires_grad=True, dtype=torch.float64)
b = torch.randn(1,2, requires_grad=True, dtype=torch.float64)
linear_test=torch.autograd.gradcheck(linear,(X,W,b))
print(linear_test)


