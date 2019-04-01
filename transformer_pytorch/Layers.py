import numpy as np
import torch
import torch.nn as nn
import math, copy, time
from torch.autograd import Variable

class Embeddings(nn.Module):
    """
    Embedding Layer

    Input:
    - d_model : the dimention of the embedding vectors. Defaultly 512 in paper.
    - vocab : the size of the vocabulary

    Output of forward:
    - result : embedding / sqrt(512)
    """
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)









class PositionalEncoding(nn.Module):
    """
    Implement the PE function.

    PE(position, 2i) = sin(position / 10000^{2i/d_model})
    PE(position,2i+1)= cos(position / 10000^{2i/d_model})

    Here we use a little trick.

    10000^{-2i/d_model} = exp(2i * (-ln(10000)/d_model))

    Input:
    - d_model : the dimention of model. Defaultly 512 in paper.
    - dropout : the probability of dropout elements. Defaultly 0.5
    - max_len : the size of buffer to store our position encoding values

    Output of forward:
    - result : Dropout(EmbeddingVector + PositionEncoding)
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)









def clones(module, N):
    """
    Produce N identical layers.
    
    Input:
    - module : create by module sequential 'Embedding'+'PositionalEncoding'
    - N : number of layers
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    """
    Mask out subsequent positions.

    e.g.
    [[0, 1, 1, 1, 1, 1],
     [0, 0, 1, 1, 1, 1],
     [0, 0, 0, 1, 1, 1],
     [0, 0, 0, 0, 1, 1],
     [0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0]
     ]

    Input:
    - size : number

    Output:
    - result : lower triangle bool matrix '0' or '1'
    """
    attn_shape = (1, size, size)
    # np.triu = Upper triangle matrix
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'

    Input:
    - query,key,value : matrix
    - mask : created by function subsequent_mask
    - dropout : dropout layer

    Output:
    - result : (context, attention)

    Q : n * d_k
    k : n * d_k
    v : n * d_k

    attention = softmax(QK^T/sqrt(d_k)) : n * n
    context = attention (dot) v : n * d_k 
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = torch.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    """
    Multi-head Atttion Layer (with mask)

    Input:
    - h : the number of parallel layers of attention
    - d_model : dimention of model
    - dropout : probability of dropout

    Output of forward:
    - result : linear output of multiheadattention layer
    """
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)









class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.

    FFN(x) = max(0, xW1+b1)W2+b2

    Input:
    - d_model : dimention of the model
    - d_ff : dimention of inner linear layer, defaultly 2048
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))









class LayerNorm(nn.Module):
    """
    Construct a layernorm module (See citation for details).
    We can also use 'torch.nn.LayerNorm' instead.

    Input:
    - features : number
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2









class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.

    Input:
    - size : number
    - dropout : the probability of Dropout
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))









class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward

    Input:
    - size : number of features
    - self_attn : Multi Head Attention layer
    - feed_forward : FFN layer

    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers

    Input:
    - layer : module. Here module is consist of module sequential 'Embedding'+'PositionalEncoding'
    - N : number of layers. Defaultly 6 in paper.
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)









class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward.

    Input:
    - size : number of features
    - self_attn : Masked Multi Head self Attention Layer
    - src_attn : Masked Multi Head context Attention Layer
    - feed_forward : FFN layer
    - dropout : the probability of dropout
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.

    Input:
    - layer : decoding layer
    - N : number of the encodelayer. Defaultly 6 in paper.
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)









class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.

    Input:
    - d_model : number of the dimention of model
    - vocab : number of vocabulary
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)









class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.

    Input:
    - encoder : Encoder System -> create by 'EncoderLayer'
    - decoder : Decoder System -> create by 'DecoderLayer'
    - src_embed : Module Sequential consist of 'Embedding' and 'PositionalEncoding'
    - tgt_embed : Module Sequential consist of 'Embedding' and 'PositionalEncoding'
    - generator : Generator Part -> create by 'Generator'
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)