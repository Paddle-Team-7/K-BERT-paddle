#import torch
import paddle.nn as nn
import paddle
import sys
import paddle.nn.functional as F

class CnnSubencoder(nn.Layer):
    def __init__(self, args, vocab_size):
        super(CnnSubencoder, self).__init__()
        self.kernel_size = args.kernel_size
        self.emb_size = args.emb_size
        self.embedding_layer = nn.Embedding(vocab_size, args.emb_size)
        self.cnn = nn.Conv2D(1, args.emb_size, (args.kernel_size, args.emb_size))

    def forward(self, ids):
        emb = self.embedding_layer(ids) # batch_size * seq_length, max_length, emb_size
        padding = paddle.zeros([emb.size(0), self.kernel_size-1, self.emb_size])
        emb = paddle.concat([padding, emb], axis=1).unsqueeze(1) # batch_size, 1, seq_length+width-1, emb_size
        conv_output = F.relu(self.cnn(emb)).squeeze(3)
        conv_output =F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)

        return conv_output
