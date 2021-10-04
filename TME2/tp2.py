import torch
from torch.autograd import Function
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm


if __name__ == '__main__':

    """
    a = torch.rand((1,10),requires_grad=True)
    b = torch.rand((1,10),requires_grad=True)
    c = a.mm(b.t())
    d = 2 * c
    c.retain_grad() # on veut conserver le gradient par rapport à c
    d.backward()  ## calcul du gradient et retropropagation jusqu’aux feuilles du graphe de calcul
    print(d.grad) # Rien : le gradient par rapport à d n’est pas conservé
    print(c.grad) # Celui-ci est conservé
    print(a.grad) ## gradient de c par rapport à a qui est une feuille
    print(b.grad) ## gradient de c par rapport à b qui est une feuille
    """
    device = 'cpu'


    data = datamaestro.prepare_dataset("edu.uci.boston")
    colnames, datax, datay = data.data()
    datax = torch.tensor(datax,dtype=torch.float)
    datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

    split_prob = 0.8
    nb = int(datax.shape[0] * split_prob)
    X_train = datax[:nb]
    y_train = datay[:nb]
    X_test = datax[nb:]
    y_test = datay[nb:]

    w = torch.randn((X_train.shape[1],1), requires_grad=True, dtype=torch.float)
    b = torch.randn(1,       requires_grad=True, dtype=torch.float)
    writer = SummaryWriter()
    #
    # print(" ############## full batch ##############")
    # max_iter = 500
    # eps = 10e-3
    #
    # for i in range(max_iter):
    #     print("epoch : ",i)
    #     yhat_train = X_train @ w + b
    #     loss_train = torch.sum(torch.pow(yhat_train - y_train, 2))
    #     loss_train.backward()
    #
    #     with torch.no_grad():
    #         w -= eps * w.grad
    #         b -= eps * b.grad
    #         w.grad.zero_()
    #         b.grad.zero_()
    #
    #
    #     writer.add_scalar('fullbatch/train', torch.mean(loss_train), max_iter)
    #
    #     yhat_test = X_test @ w + b
    #     loss_test = torch.sum(torch.pow(yhat_test - y_test, 2))
    #     writer.add_scalar('fullbatch/test', torch.mean(loss_test), max_iter)

    #
    # print(" ############## SGD ##############")
    # max_iter = 500
    # eps = 10e-3
    #
    # for i in range(max_iter):
    #     print("epoch : ",i)
    #     ind = torch.randperm(X_train.shape[0])
    #     X = X_train[ind]
    #     y = y_train[ind]
    #     for j in range(X_train.shape[0]):
    #         yhat_train = X[j] @ w + b
    #         loss_train = torch.sum(torch.pow(yhat_train - y[j], 2))
    #         loss_train.backward()
    #
    #         with torch.no_grad():
    #             w -= eps * w.grad
    #             b -= eps * b.grad
    #             w.grad.zero_()
    #             b.grad.zero_()
    #
    #
    #     yhat_train = X_train @ w + b
    #     loss_train = torch.sum(torch.pow(yhat_train - y_train, 2))
    #     writer.add_scalar('SGD/train', torch.mean(loss_train), max_iter)
    #
    #     yhat_test = X_test @ w + b
    #     loss_test = torch.sum(torch.pow(yhat_test - y_test, 2))
    #     writer.add_scalar('SGD/test', torch.mean(loss_test), max_iter)
    #

    print(" ############## minibatch ##############")
    max_iter = 500
    eps = 10e-3
    batch_size = 32

    for i in range(max_iter):
        print("epoch : ",i)
        ind = torch.randperm(X_train.shape[0])
        X = X_train[ind]
        y = y_train[ind]
        for j in range(0,X.shape[0], batch_size):
            indices = ind[j:j + batch_size]
            X, y = X_train[indices], y_train[indices]
            yhat_train = X @ w + b
            loss_train = torch.sum(torch.pow(yhat_train - y, 2))
            loss_train.backward()

            with torch.no_grad():
                w -= eps * w.grad
                b -= eps * b.grad
                w.grad.zero_()
                b.grad.zero_()


        yhat_train = X_train @ w + b
        loss_train = torch.sum(torch.pow(yhat_train - y_train, 2))
        writer.add_scalar('minibatch/train', torch.mean(loss_train), max_iter)

        yhat_test = X_test @ w + b
        loss_test = torch.sum(torch.pow(yhat_test - y_test, 2))
        writer.add_scalar('minibatch/test', torch.mean(loss_test), max_iter)


    # # Descente de gradient test:
    #
    # X = torch.randn((5,10),requires_grad=True,dtype=torch.float,device=device)
    # y = torch.randint(5,size=(5,),dtype=type,device=device)
    # w = torch.randn((1,10),requires_grad=True,dtype=torch.float,device=device)
    # print(y)
    #
    # eps = 10e-3
    # iter = 500
    # for i in range(iter):
    #
    #     yhat = (X@w.T).reshape(-1,)
    #     mse = torch.sum(torch.pow(yhat-y,2))
    #     mse.backward()
    #
    #
    #     print("yhat :",yhat," y_true :",y, "Loss : ",mse)
    #     with torch.no_grad():
    #         w -=  eps*w.grad
    #         w.grad.zero_()

    # TODO:
