# -*- encoding:utf-8 -*-
import paddle
import paddle.nn as nn


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = paddle.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long)
    return x[tuple(indices)]