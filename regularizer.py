import torch

def sparse_l1(loss,inputs):
    grad_params = torch.autograd.grad(loss, inputs, create_graph=True, allow_unused=True)
    regval = (grad_params[0].norm(1))
    return regval