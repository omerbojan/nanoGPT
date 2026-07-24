from collections import defaultdict
import math
from pathlib import Path
import torch
import torch.nn.functional as F

from config.mytrain_shakespeare_char import *
from data.shakespeare_char.myprepare import load_data
from mymodel import Transformer

tokenizer, Xtr, Ytr, Xevl, Yevl = load_data()
dvocub = tokenizer.dvocub
iterations = 1000
lre = torch.linspace(-8,-3,iterations)
lri = 2**lre
lri = [2**(-7) for i in range(iterations)]
lossi = []
params = defaultdict(list)
tommy = Transformer(dvocub, dcontext, dheads, dembedding, dcorrelation, dhidden, dblock)

currper = 0.0
optimizer = torch.optim.AdamW(tommy.parameters(), lr=3e-4)
for i in range(iterations):
    
    ix = torch.randint(0, len(Xtr) - dcontext + 1, (batchsize,))
    ix = ix.unsqueeze(1) + torch.arange(dcontext)#shape (batch,1) + (context,1)
    xtr = Xtr[ix]
    ytr = Ytr[ix]
    y = tommy(xtr)
    #y now has shape batch,context,dvocub
    #ytr has shape batch,context
    loss = F.cross_entropy(y.flatten(0,1), ytr.flatten(0,1))
    lossi.append(loss.item())

    #Now we gradient
    # for p in tommy.parameters():
    #     p.grad = None
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # with torch.no_grad():
    #     for p in tommy.parameters():
    #         p -= p.grad * lri[i]
    newcurrper = float(i)/iterations
    if newcurrper > currper+1./100:
        print(newcurrper)
        currper = newcurrper
print(f"Initial losss should be {math.log(dvocub)}")
torch.save({"model": tommy.state_dict(), "config": {
    "dvocub": dvocub, "context": dcontext, "dheads": dheads,
    "dembedding": dembedding, "dcorrelation": dcorrelation,
    "dhidden": dhidden, "blocks": dblock,
}}, Path(__file__).with_name("mycheckpoint.pt"))
