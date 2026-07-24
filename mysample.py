from config.mytrain_shakespeare_char import *
from mymodel import Transformer
import torch


outputsize=100
T = 1
currcontext = [0]
final = currcontext[:]
for i in range(outputsize):
    input = torch.tensor(currcontext).unsqueeze(0) #adds fake batch dim
    newprobs = tommy(input)[0][-1].softmax(0)
    newbyte = torch.multinomial(newprobs,num_samples=1).item()
    currcontext.append(newbyte)
    final.append(newbyte)
    if len(currcontext) > dcontext:
        currcontext= currcontext[1:]

