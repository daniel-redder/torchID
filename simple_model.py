import torch
import torch.nn.init as init


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        
        self.l1 = torch.nn.Linear(4, 1)

        init.uniform_(self.l1.weight, -0.1, 0.1)
        init.uniform_(self.l1.bias, -0.1, 0.1)

    def forward(self, x):
        
        x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-12, out=None)
        assert torch.isnan(x).sum() == 0, f"Nan in x, {x}"
        x = self.l1(x)

        return x
    


