import torch
import torch.nn as nn
from typing import Dict, Tuple

class XORModel(nn.Module):
    def __init__(self, hidden_dim: int = 4): # входной слой 2 нейрона, скрытый N, выходной 1.
        super().__init__()
        self.input_dim = 2
        self.hidden_dim = hidden_dim
        self.output_dim = 1

        self.layer_hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.act_hidden = nn.Tanh()

        self.layer_output = nn.Linear(self.hidden_dim, self.output_dim)
        self.act_output = nn.Sigmoid()



    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z1 = self.layer_hidden(x)
        a1 = self.act_hidden(z1)
        z2 = self.layer_output(a1)
        a2 = self.act_output(z2)
        return {
            "inputs": x,
            "hidden_pre": z1,
            "hidden_post": a1,
            "output_pre": z2,
            "output_post": a2
        }


    