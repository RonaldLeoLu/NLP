# word embedding models
import torch
import torch.nn as nn
import numpy as np

class HierSoft_CBOW(nn.Module):
    def __init__(self, nwords, emb_size, hier_nodes, batch_size=1):
        '''
        Input:
        - nwords : the number of words that we need to embedde
        - emb_size : the demision of word embedding
        - hier_nodes : the number of non-leaf nodes
        - batch_size : size of batch
        '''
        super(HierSoft_CBOW, self).__init__()
        # Embedding Layer
        self.u_emblayer = nn.Embedding(nwords, emb_size)
        nn.init.xavier_normal(self.u_emblayer.weight)
        self.v_emblayer = nn.Embedding(hier_nodes, emb_size)
        nn.init.xavier_normal(self.v_emblayer.weight)

    def forward(self, words, h_code, h_path):
        '''
        Input:
        - words : the idx of context words in windows
        - h_code : the huffman code of center word
        - h_path : the huffman path of center word
        '''
        embs = self.u_emblayer(words)
        # 1xN
        xw = torch.sum(embs, dim=0).view(1,-1)
        #print('xw shape: {}'.format(xw.shape))
        # NxR
        theta = self.v_emblayer(h_path).t()
        #print('theta shape: {}'.format(theta.shape))
        # Rx1
        hcode = torch.FloatTensor(h_code).view(-1,1)
        # 1xR
        z = torch.sigmoid(torch.matmul(xw,theta))
        #print('z is {}'.format(z))

        return torch.matmul(torch.log(z), hcode) + torch.matmul(torch.log(1-z), 1-hcode)


class NegSpl_SG(nn.Module):
    def __init__(self, nwords=0, emb_size=0, neg_size=0, weights = None, batch_size=1):
        super(HierSoft_CBOW, self).__init__()
        # Embedding Layer
        self.u_emblayer = nn.Embedding(nwords, emb_size)
        nn.init.xavier_normal(self.u_emblayer.weight)
        self.v_emblayer = nn.Embedding(nwords, emb_size)
        nn.init.xavier_normal(self.v_emblayer.weight)
        # weights
        self.weights = weights
        if self.weights is not None:
            wt = np.power(self.weights, 0.75)
            wt = wt / wt.sum()
        self.weights = wt
        self.neg_size = neg_size

    def sampling(self, widx):
        current_wgt = self.weights.copy()
        current_wgt[widx] = 0
        return torch.multinomial(torch.FloatTensor(current_wgt), self.neg_size)

    def forward(self, c_word, context):
        xw = self.u_emblayer(c_word)
        xs = self.v_emblayer(context).t()
        xneg = self.u_emblayer(self.sampling(c_word))

        pos = torch.log(torch.Sigmoid(torch.matmul(xw, xs))).sum()
        neg = torch.log(1 - torch.Sigmoid(torch.matmul(xneg, xs))).sum()

        return pos+neg