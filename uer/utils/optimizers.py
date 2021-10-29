# -*- encoding:utf-8 -*-
"""
The optimizer is identical with huggingface's implementation.
See https://github.com/huggingface/pytorch-pretrained-BERT
"""

import math
#import torch
import paddle
from paddle.optimizer import Adam
#from torch.optim import Optimizer
from paddle.optimizer import Optimizer
from collections import defaultdict
#from torch.nn.utils import clip_grad_norm_

def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
        return 0.5 * (1.0 + paddle.cos(math.pi * x))
    #return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}


class BertAdam(Optimizer):
    """
    Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self,
                learning_rate,
                beta1=0.9, 
                beta2=0.999,
                epsilon=1e-6,
                parameters=None, 
                weight_decay=None,
                grad_clip=None,
                name=None,
                lazy_mode=False,
                warmup=-1, 
                t_total=-1,
                max_grad_norm=1.0,
                schedule='warmup_linear',
                weight_decay_rate = 0.01):

        if not learning_rate >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(learning_rate))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(beta2))
        if not epsilon >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(epsilon))

        self.state = defaultdict(dict)

        super(BertAdam, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip = grad_clip,
            name = name)
        
        self.type = 'bertadam'
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon
        self.weight_decay_rate = weight_decay_rate
        self.t_total = t_total
        self.warmup = warmup
        self.schedule = schedule
        self.learning_rate = learning_rate

    def get_lr(self):
        lr = []
        for param in self._parameter_list:
            state = self.state[p]
            if len(state) == 0:
                return [0]
            if self.t_total != -1:
                schedule_fct = SCHEDULES[self.schedule]
                lr_scheduled = self.learning_rate * schedule_fct(state['step']/self.t_total, self.warmup)
            else:
                lr_scheduled = self.learning_rate
            lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for param in self._parameter_list:
            if param.stop_gradient:
                continue
            if param._grad_ivar() is not None:  # p = param
                grad_var = param._grad_ivar()
                if hasattr(grad_var, "_is_sparse") and grad_var._is_sparse() and self.regularization is not None:
                    raise RuntimeError(
                        "Adam don't support weight_decay with sparse parameters, please set it to None.")
                #params_grads.append((param, grad_var))

            state = self.state[param]
            # state initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                #state['next_m'] = torch.zeros_like(p.data)
                state['next_m'] = paddle.zeros_like(paddle.to_tensor(param))
                # Exponential moving average of squared gradient values
                #state['next_v'] = torch.zeros_like(p.data)
                state['next_v'] = paddle.zeros_like(param)

            next_m, next_v = state['next_m'], state['next_v']
            beta1, beta2 = self.beta1, self.beta2

            # Add grad clipping
            if self.max_grad_norm > 0:
                clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=self.max_grad_norm)
                self._grad_clip = clip
            
            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            #next_m.mul_(beta1).add_(1 - beta1, grad)
            xa = paddle.multiply(next_m,paddle.to_tensor(beta1))
            next_m = paddle.add(xa,paddle.to_tensor(1 - beta1), grad_var)
            #next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad) # grad*grad*(1 - beta2)+xa
            xa = paddle.multiply(next_v,paddle.to_tensor(beta2))
            #next_v = paddle.addmm(xa, grad_var, grad_var, alpha=paddle.to_tensor(1 - beta2), beta=1.0)
            next_v = xa + grad_var * paddle.to_tensor(1 - beta2) * grad_var

            update = next_m / (next_v.sqrt() + self.epsilon)

            if self.weight_decay_rate > 0.0:
                update += self.weight_decay_rate * param.detach()
            
            if self.t_total != 1:
                schedule_fct = SCHEDULES[self.schedule]
                lr_scheduled = self.learning_rate * schedule_fct(state['step']/self.t_total, self.warmup)
            else:
                lr_scheduled = self.learning_rate

            update_with_lr = lr_scheduled * update
            #param.data.add_(-update_with_lr)
            param = param + -update_with_lr

            state['step'] += 1

        return loss

