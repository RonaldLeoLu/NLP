import torch
import torch.nn as nn

import numpy as np

from build_model import make_model
from run_model import *

# Synthetic Data

def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10))).long()
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)

# Loss Computation

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        #print('Loss:',loss.data.numpy())
        #print('Type loss:', type(loss))
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm

# This code predicts a translation using greedy decoding for simplicity.

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


if __name__ == '__main__':
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model, 
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5), model, 
                        SimpleLossCompute(model.generator, criterion, None)))

    model.eval()
    src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
    src_mask = Variable(torch.ones(1, 1, 10) )
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))