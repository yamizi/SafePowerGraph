
import torch
from torch.nn import functional as F
from torch import Tensor

class AdvLoss(torch.nn.MSELoss):
    def forward(self, input: Tensor, target: Tensor, idx_attack) -> Tensor:

        mask = ~torch.isnan(target)
        return F.mse_loss(input[mask], target[mask], reduction=self.reduction)

