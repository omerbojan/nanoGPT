import torch
import torch.nn as nn

#-----------------------------------------------------------------------------------
class Embedding(nn.Module):
    """
    Embeds vocub in the linear space
    """
    def __init__(self, dvocub, dembedding):
        
        super().__init__()
        self.dvocub = dvocub
        self.dembedding = dembedding
        self.embed = nn.Embedding(dvocub, dembedding)
    
    def forward(self,x):
        return self.embed(x)
    
#------------------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    """
    Just holds the vector offset we add
        """
    def __init__(self, dcontext, dembedding):
        super().__init__()
        self.offset = nn.Parameter(torch.randn(dcontext, dembedding))
    
    def forward(self,x):
        #x has shape batch,context,dembedding
        return self.offset[:x.shape[1]]
    

"""
As a prep for the transformer block, let's repeat how it works


tokens->vocub->embeddings->embed+position->(attention_heads= xQ is the query matrix, xK is the key matrix. Their rows
are the questions/keys. Then (xQ)(xK)^t has (i,j) how xj answers xi. Then if j>i we mask, so we get upper triangular.
Then we concanate, then layer norm, then add to x. So

just does the heads part

Parameters-
context
dembedding 
dcorrelation
    Q,K will be (embed,correlation)
dhidden
    V will be (correlation, dhidden)
"""
#-------------------------------------------------------------------------------------
class AttentionHead(nn.Module):
    def __init__(self, dembedding, dcorrelation,dhidden, name="AttentionHead"):
        super().__init__()
        self.name = name
        self.dembedding = dembedding
        self.dcorrelation = dcorrelation #this is the correlation_space
        self.dhidden = dhidden
        self.Q = nn.Linear(dembedding, dcorrelation)
        self.K = nn.Linear(dembedding, dcorrelation)
        self.V = nn.Linear(dembedding,dhidden)

    def forward(self,x):
        """
        x.shape = (batch,context, dembedding)
        """
        assert(x.shape[2] == self.dembedding)
        context = x.shape[1]
        xQ = self.Q(x)
        xK = self.K(x)
        xV = self.V(x)
        #these have shape 
        #in training: (batch, context, dcorrelation\dhidden)
        #in inference:
        #b = batch, c,d = context, k = korrelation
        xQxKt = torch.einsum("bck,bdk->bcd", xQ, xK)
        nxQxKt = xQxKt * self.dcorrelation**(-0.5)
        upper_triangular = torch.triu(
            torch.full((context, context), float("-inf"), device=x.device), diagonal=1
        )
        bupper_triangular = upper_triangular.unsqueeze(0)
        masked_xQxKt = nxQxKt + bupper_triangular #broadcast
        probs = masked_xQxKt.softmax(dim=-1)
        #shape is (Batch,context,context) with (i,j) is how j answers i
        changes = torch.einsum("bcd,bdh->bch", probs, xV)
        #if self.trianing: batch, context, hidden
        #if inference: context, hidden
        return changes
        
"""
Parameters-
context
heads
dembedding
dcorrelation
dhidden

x=x+(projection(heads(layer(x)))).
x = x+(ffn(layer(x))).
Here both layer, ffn are per token-vector
"""
#-------------------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, context, dheads, dembedding, dcorrelation, dhidden):
        super().__init__()

        self.context = context
        self.dheads = dheads
        self.dembedding= dembedding
        self.dcorrelation = dcorrelation
        self.dhidden = dhidden

        self.LN1 = nn.LayerNorm(dembedding)
        self.heads = nn.ModuleList()
        for i in range(dheads):
            head = AttentionHead(dembedding,dcorrelation, dhidden)
            self.heads.append(head)
        self.projection = nn.Linear(dhidden*dheads , self.dembedding)

        self.ffn = nn.Sequential(nn.LayerNorm(dembedding), nn.Linear(dembedding, 4*dembedding), nn.Tanh(), nn.Linear(4*dembedding,dembedding))
    
    def forward(self,x):
        #in training, x.shape = batch, context, dembed
        #in inference, x = context, dembed
        batch = x.shape[0]
        xcontext= x.shape[1]
        LN1x = self.LN1(x)
        heads_x = torch.stack([head(LN1x) for head in self.heads],dim=-1)
        #in training each of those head(LN1x) has shape batch,context,hidden,heads
        heads_x = heads_x.view(batch, xcontext, self.dhidden*self.dheads)
        x = x + self.projection(heads_x)
        #in training has shape batch, context, dembedding

        x = x + self.ffn(x)
        #in training has shape batch, context, dembedding
        return x
    
#------------------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, dvocub, context, dheads, dembedding, dcorrelation, dhidden, blocks = 1):
        super().__init__()
        self.dvocub, self.context, self.dheads, self.dembedding, self.dcorrelation, self.dhidden, self.blocks = dvocub, context, dheads, dembedding, dcorrelation, dhidden, blocks

        self.embedding = Embedding(dvocub, dembedding)
        self.position_shift = PositionalEmbedding(context, dembedding)
        self.blocks = nn.ModuleList([TransformerBlock(context, dheads, dembedding, dcorrelation, dhidden) for i in range(blocks)])
        self.LN3 = nn.LayerNorm(self.dembedding)
        self.to_logits = nn.Linear(self.dembedding, self.dvocub)

    def forward(self, w):
        """
        x= embed(w)
        x = x + positional(x)
        for each block:
            x = block(x)
        (that block includes residula as well)
        
        now we grab the last vector, layernorm, and ffn into dvocub
        """
        x = self.embedding(w)
        x = x + self.position_shift(x)

        for b in self.blocks:
            x = b(x)
            #x.shape = batch, context, embedding
            #we now care about all outputs
        x = self.LN3(x)
        #Now the shape is (batch, context), embedding
        logits = self.to_logits(x)
        #Now the shape is (batch, context), dvocub
        return logits

    def probs(self,w):
        logits = self(w)
        return logits.softmax(dim=-1)
