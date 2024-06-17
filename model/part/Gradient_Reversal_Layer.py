from typing import Any, Optional, Tuple
from torch.autograd import Function
import torch.nn as nn
import torch

"""
http://sites.skoltech.ru/compvision/projects/grl/files/paper.pdf
"""


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GRL(nn.Module):
    def __init__(self, coeff: Optional[float] = 1.):
        super(GRL, self).__init__()
        self.coeff = coeff

    def forward(self, *input):
        return GradientReverseFunction.apply(*input, self.coeff)



