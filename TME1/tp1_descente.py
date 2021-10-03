import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context


# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3)
b = torch.randn(3)

epsilon = 0.001

writer = SummaryWriter()

l=Linear()
mse=MSE()
list_loss=[]
x=[]
for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)
    ctx=Context()
    ctx_loss=Context()
    yhat=l.forward(ctx,x,w,b)
    loss=mse.forward(ctx_loss,yhat,y)
    yhat_grad,y_grad=mse.backward(ctx_loss,1)
    x_grad,w_grad,b_grad = l.backward(ctx,yhat_grad)
    w=w-epsilon*w_grad
    b=b-b_grad*epsilon
    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")


print(w)
print(b)
