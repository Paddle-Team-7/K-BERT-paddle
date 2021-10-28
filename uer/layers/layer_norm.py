# -*- encoding:utf-8 -*-
#import torch
#import torch.nn as nn
import paddle
import paddle.nn as nn


class LayerNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        #self.gamma = nn.Parameter(paddle.ones([hidden_size])) #requires_grad=False
        x = paddle.ones([hidden_size])
        self.gamma = paddle.create_parameter(shape=x.shape,dtype=str(x.numpy().dtype),
                                default_initializer=paddle.nn.initializer.Assign(x))
        self.gamma.stop_gradient = True

        #self.beta = nn.Parameter(paddle.zeros([hidden_size]))
        x = paddle.zeros([hidden_size])
        self.beta = paddle.create_parameter(shape=x.shape,dtype=str(x.numpy().dtype),
                                default_initializer=paddle.nn.initializer.Assign(x))
        self.beta.stop_gradient = True

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x-mean) / (std+self.eps) + self.beta
