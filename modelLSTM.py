import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import GenerateInputs

useGPU = False

class modelLSTM(nn.Module):

    def __init__(self,input_size, hidden_dim, num_layers):
        super(modelLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_dim, num_layers = num_layers)
        self.targetLayer = nn.Linear(hidden_dim, 1)
        self.featureLayer = nn.Linear(hidden_dim, 1)


    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if useGPU:
            return (autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim).cuda()),
                    autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim).cuda()))
        else:
            return (autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)),
                    autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)))

    def forward(self, input):
        for i in input:
            lstm_out, self.hidden = self.lstm(i.view(1, 1, -1), self.hidden)
        targetOut = self.targetLayer(lstm_out)
        featureOut = self.featureLayer(lstm_out)
        return targetOut,featureOut

    def custom_loss(self,x,xPred,y,yPred,alpha):
        return alpha*(x-xPred)**2 + (1-alpha)*(y-yPred)**2



if useGPU:
    model = modelLSTM(1,32,2).cuda()
else:
    model = modelLSTM(1, 32, 2)
inputs,targets = GenerateInputs.getInputs()
optimizer = optim.SGD(model.parameters(), lr=0.1)


def getNextVal(inputs,i):
    j = 0
    while j < len(inputs[i]):
        if inputs[i][j] == 0:
            break
        j += 1
    if j == 80 or i == len(inputs)-1:
        return inputs[i][j-1]
    return inputs[i+1][j]

def train():
    for epoch in range(100):
        i= 0
        epochLoss = 0
        while i < len(inputs):

            if useGPU:
                input1 = autograd.Variable(torch.FloatTensor(inputs[i]).cuda())
            else:
                input1 = autograd.Variable(torch.FloatTensor(inputs[i]))
            input1 = input1.view(80, 1, -1)
            if useGPU:
                y = autograd.Variable(torch.FloatTensor(targets[i]).cuda())
            else:
                y = autograd.Variable(torch.FloatTensor(targets[i]))

            model.hidden = model.init_hidden()
            model.zero_grad()

            yhat,xhat = model.forward(input1)
            x = getNextVal(inputs, i)
            if useGPU:
                x = autograd.Variable(torch.FloatTensor([x]).cuda())
            else:
                x = autograd.Variable(torch.FloatTensor([x]))

            loss = model.custom_loss(x,xhat,y,yhat,0.5)
            epochLoss += float(loss.data[0])
            loss.backward()
            optimizer.step()
            del input1
            del y
            del x

            i = i+1
        print("epoch #"+str(epoch)+" loss = "+str(epochLoss/len(inputs)))


train()