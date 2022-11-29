import torch
import mindspore as ms
import numpy as np
from torch import  optim

np.random.seed(0)

def pytrain(epoch: int,ptmodel,input):
    label = np.random.randn(1, 3, 224, 224).astype(np.float32)
    label = torch.tensor(label)
    loss = torch.nn.MSELoss()
    optimizer = optim.SGD(ptmodel.parameters(), lr=0.03)
    for i in range(epoch):
            output = ptmodel(input)
            l = loss(output, label)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

def mstrain(epoch: int,msmodel,input):
    label = np.random.randn(1, 3, 224, 224).astype(np.float32)
    label = ms.Tensor(label)

    loss_fn = ms.nn.MSELoss()
    optim = ms.nn.SGD(msmodel.trainable_params(), learning_rate=0.03)

    loss_net = ms.nn.WithLossCell(msmodel, loss_fn)
    train_net = ms.nn.TrainOneStepCell(loss_net, optim)
    train_net.set_train()
    for i in range(epoch):
            train_net(input, label)