import torch
import math
from Play import Play

maxTimes=100000
BatchSize=1000

matrixSize=8
layers=200
features=512

Play_lr=2e-4
Adam_lr=2e-4
weight_decay=0


class Block(torch.nn.Module):
    def __init__(self,in_features,out_features,gain=1,**kwargs):
        super(Block, self).__init__()
        self.Linear=torch.nn.Linear(in_features,out_features,bias=True,**kwargs)
        with torch.no_grad():
            torch.nn.init.orthogonal_(self.Linear.weight,gain)
            self.Linear.bias.fill_(0)
    def forward(self,input):
        return self.Linear(input).asinh()

#This network architecture is also my original work and follows the same open source license,
#including its extension to any/every dimensional convolution and replacement of any/every activation functions.        
class AccessNet(torch.nn.Module):
    def __init__(self,in_features,out_features,hidden_layers,hidden_features,**kwargs):
        super(AccessNet, self).__init__()
        self.inLayer=torch.nn.Linear(in_features,hidden_features,bias=True,**kwargs)
        self.Blocks=torch.nn.Sequential() 
        for t in range(hidden_layers):
            self.Blocks.add_module('hiddenLayer{0}'.format(t),Block(hidden_features,hidden_features,**kwargs))
        self.outLayer=torch.nn.Linear(hidden_features,out_features,bias=True,**kwargs)
        with torch.no_grad():
            torch.nn.init.orthogonal_(self.inLayer.weight,1)
            self.inLayer.bias.fill_(0)
            torch.nn.init.orthogonal_(self.outLayer.weight,1/math.sqrt(hidden_layers))
            self.outLayer.bias.fill_(0)
    def forward(self,input):
        step=self.inLayer(input)
        outputAccess=torch.zeros_like(step)
        for block in self.Blocks:
            step=block(step)
            outputAccess=outputAccess+step
        output=self.outLayer(outputAccess)
        return output

device = "cuda" if torch.cuda.is_available() else "cpu"

moduel1=AccessNet(matrixSize*matrixSize,matrixSize*matrixSize,layers,features,device=device)
moduel2=AccessNet(matrixSize*matrixSize,matrixSize*matrixSize,layers,features,device=device)
optimizer1 = torch.optim.RAdam(moduel1.parameters(),Adam_lr,weight_decay=weight_decay)
optimizer2 = Play(moduel2.parameters(),lr=Play_lr,weight_decay=weight_decay)

eye=torch.eye(matrixSize,device=device).unsqueeze(0)
eye=eye/eye.std()

import visdom
wind = visdom.Visdom(env="Optimizer Test", use_incoming_socket=False)
    
wind.line([[float('nan'),float('nan')]],[0],win = 'loss',opts = dict(title = 'log(loss)/log(batchs)',legend = ['RAdam','Play']))

print(moduel1)
    
for time in range(maxTimes):
    moduel1.zero_grad()
    input=torch.rand(BatchSize,matrixSize*matrixSize,device=device)*2-1
    output=moduel1(input)
    input=input.view(BatchSize,matrixSize,matrixSize)
    output=output.view(BatchSize,matrixSize,matrixSize)
    e=input@output
    e=e/e.std()
    loss1=(e-eye).abs().mean()
    loss1.backward()
    optimizer1.step()
    
    moduel2.zero_grad()
    input=input.view(BatchSize,matrixSize*matrixSize)
    output=moduel2(input)
    input=input.view(BatchSize,matrixSize,matrixSize)
    output=output.view(BatchSize,matrixSize,matrixSize)
    e=input@output
    e=e/e.std()
    loss2=(e-eye).abs().mean()
    loss2.backward()
    optimizer2.step(loss=loss2)
    
    wind.line([[float(loss1.log()),float(loss2.log())]],[math.log(time+1)],win = 'loss',update = 'append')
    
