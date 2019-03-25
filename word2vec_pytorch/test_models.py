# test code
import torch
import random
from collections import defaultdict
from tqdm import tqdm

from load_data import get_HuffmanCodePath, read_dataset
from model import HierSoft_CBOW

nodes, hcodes, hpath = get_HuffmanCodePath('ptb')
#print('All Tree nodes is %d'%nodes[0])

w2i = defaultdict(lambda : len(w2i))

train = list(read_dataset(w2i, 'ptb'))[:2]
i2w = {v:k for k,v in w2i.items()}


nwords = len(i2w)


EMB_SIZE = 20
ITERS = 10
WIN_SIZE = 2

model = HierSoft_CBOW(nwords, EMB_SIZE, nodes[0]+1)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

data_type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    data_type = torch.cuda.LongTensor
    model.cuda()

def calc_sent_loss(sents):
    paddle_sent = [w2i['<unk>']] * WIN_SIZE + sent + [w2i['<unk>']] * WIN_SIZE

    losses = []
    for i in range(WIN_SIZE, len(sents) + WIN_SIZE):
        c = torch.Tensor(paddle_sent[i-WIN_SIZE : i] + paddle_sent[i+1 : i+WIN_SIZE+1]).type(data_type)
        t = i2w[sents[i - WIN_SIZE]]
        tcode = hcodes[t]
        tpath = torch.Tensor(hpath[t]).type(data_type)
        #print('this center word path is {}'.format(tpath))
        loss = model.forward(c, tcode, tpath)
        #print('loss is {}'.format(loss))
        losses.append(loss)

    return torch.stack(losses).sum()


for i in range(ITERS):
    random.shuffle(train)
    train_words, train_loss = 0, 0.0

    for sent_id, sent in tqdm(enumerate(train)):
        my_loss = calc_sent_loss(sent)
        train_loss += my_loss.item()
        train_words += len(sent)

        opt.zero_grad()
        my_loss.backward()
        opt.step()

    print('The {}th iters get loss of {}'.format(i, train_loss))


