from typing import Dict
import torch
import torch.nn as nn

class UncertaintyWeighter(nn.Module):
    """
    Homoscedastic uncertainty weighting per multitask:
    L = sum_t [ exp(-s_t) * L_t + 0.5 * s_t ], con s_t = log(sigma_t^2).
    """
    def __init__(self, task_names, init_log_var: float = 0.0):
        super().__init__()
        self.task_names = list(task_names)
        self.log_vars = nn.ParameterDict({
            t: nn.Parameter(torch.tensor(float(init_log_var)))
            for t in self.task_names
        })

    @torch.no_grad()
    def current_weights(self) -> dict:
        return {t: float(torch.exp(-p)) for t, p in self.log_vars.items()}

    def forward(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for t, Lt in loss_dict.items():
            if Lt.dim() > 0:
                Lt = Lt.mean()
            s_t = self.log_vars[t]
            total = total + torch.exp(-s_t) * Lt + 0.5 * s_t
        return total