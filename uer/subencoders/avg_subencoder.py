#import torch
import paddle
import paddle.nn as nn
import sys
import paddle.nn.functional as F

class AvgSubencoder(nn.Layer):
    def __init__(self, args, vocab_size):
        super(AvgSubencoder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, args.emb_size)

    def forward(self, ids):
        emb = self.embedding_layer(ids) # batch_size, max_length, emb_size
        output = emb.mean(1)
        return output
