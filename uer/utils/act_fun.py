# -*- encoding:utf-8 -*-
import math
import paddle

def gelu(x):
    return x * 0.5 * (1.0 + paddle.erf(x / math.sqrt(2.0)))