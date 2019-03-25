import torch
import torch.nn as nn

class HiearchSoft(nn.Module):
    def __init__(self, nnodes=5, n_embs=None):
        super().__init__()
        '''huffman path'''
        self.seqs = nn.ModuleList()
        for i in range(nnodes):
            seq = nn.Linear(n_embs, 1, bias=False)
            self.seqs.append(seq)

    def forward(self, xw, h_code, h_path):
        losses = []
        for idx in range(len(h_path)):
            z = self.seqs[h_path[idx]](xw)
            sigz = torch.sigmoid(z)
            p = h_code[idx] * torch.log(sigz) + \
                (1 - h_code[idx]) * torch.log(1 - sigz)
            losses.append(p)
        return sum(losses)


if __name__ == '__main__':
    mymode = HiearchSoft(n_embs=5)
    ori_para = list(mymode.parameters())
    xw = torch.ones(1,5, requires_grad=True)
    print('Old xw:',xw)
    for i in range(len(ori_para)):
        print('Old params idx %d'%i, ori_para[i])
    opt = torch.optim.Adam([{'params':xw}, {'params':mymode.parameters()}],lr=2.3)
    opt.zero_grad()
    hcode = [0,1]
    h_path = [0,2]
    loss = mymode.forward(xw, hcode, h_path)
    loss.backward()
    opt.step()
    new_para = list(mymode.parameters())
    print('='*20)
    print('New xw:',xw)
    for j in range(len(new_para)):
        print('New params idx %d'%j, new_para[j])