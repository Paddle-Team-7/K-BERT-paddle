# -*- encoding:utf-8 -*-
#import torch
#import torch.nn as nn
import paddle
import paddle.nn as nn
from uer.layers.layer_norm import LayerNorm
import numpy as np

class BertEmbedding(nn.Layer):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """
    def __init__(self, args, vocab_size):
        super(BertEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)

        # # # 给embedding赋权重
        # self.word_embedding.weight.set_value(np.load("word_embedding_weight.npy"))
        # print("self.word_embedding.weight")
        # print(self.word_embedding.weight)
        # self.position_embedding.weight.set_value(np.load("position_embedding_weight.npy")) 
        # self.segment_embedding.weight.set_value(np.load("segment_embedding_weight.npy"))
        
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg, pos=None):
        word_emb = self.word_embedding(src)
        if pos is None:
            pos_emb = self.position_embedding(paddle.arange(0, word_emb.size(1), dtype='int64').unsqueeze(0).repeat(word_emb.size(0), 1))
        else:
            pos_emb = self.position_embedding(pos)
        seg_emb = self.segment_embedding(seg)

        emb = word_emb + pos_emb + seg_emb
        
        emb = self.dropout(self.layer_norm(emb))
        #emb = self.layer_norm(emb)
        return emb

