# -*- encoding:utf-8 -*-
#import torch
import paddle
import collections


def save_model(model, model_path):
    # We dont't need prefix "module".
    if hasattr(model, "module"):
        paddle.save(model.module.state_dict(), model_path)
    else:
        paddle.save(model.state_dict(), model_path)
