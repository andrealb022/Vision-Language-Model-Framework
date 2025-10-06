from typing import Dict
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt

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

class RunningMeans:
    """
    Tracks the exponential moving average (EMA) of loss values for multiple tasks.
    Each task's running mean is updated at every call to `update()` using the 
    corresponding loss value. The first loss value initializes the running mean. 
    Useful for smoothing noisy loss curves and computing task balancing weights.
    """
    def __init__(self, task_names, alpha=0.99):
        """
        Args:
            task_names (list of str): Names of the tasks in order.
            alpha (float): EMA smoothing factor (closer to 1 -> smoother).
        """
        self.task_names = list(task_names)
        self.alpha = float(alpha)
        self.values = {t: None for t in self.task_names}  # Current EMA values
        self.history = {t: [] for t in self.task_names}   # History for plotting

    def update(self, losses):
        """
        Update EMA values for all tasks.

        Args:
            losses (list of float): Loss values for each task (same order as task_names).
        """
        for idx, task in enumerate(self.task_names):
            current = losses[idx]
            if self.values[task] is None:  # First update sets initial value
                new_val = current
            else:
                old_val = self.values[task]
                new_val = self.alpha * old_val + (1 - self.alpha) * current
            self.values[task] = new_val
            self.history[task].append(new_val)
    
    def update_by_idx(self, loss_value, task_idx):
        """
        Update EMA for a single task.

        Args:
            loss_value (float): e.g., loss_tensor.item()
            task_idx (int): index of the task in task_names
        """
        task = self.task_names[task_idx]
        v = self.values[task]
        if v is None:             # first observation initializes
            new_v = loss_value
        else:                      # standard EMA
            new_v = self.alpha * v + (1 - self.alpha) * loss_value
        self.values[task] = new_v
        self.history[task].append(new_v)

    def get(self, task_name):
        """Return the current EMA value for a given task name."""
        return self.values.get(task_name, None)

    def get_by_index(self, idx):
        """Return the current EMA value for a task by index."""
        return self.values[self.task_names[idx]]

    def plot(self, output_path=None):
        """Plot EMA history for each task."""
        plt.figure(figsize=(10,6))
        for task in self.task_names:
            plt.plot(self.history[task], label=task)
        plt.xlabel("Epoch / Iterations")
        plt.ylabel("Running Mean Loss")
        plt.title("Running Means per Task Over Time")
        plt.legend()
        plt.grid(True)
        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def save_history(self, filepath):
        """Save EMA history to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_history(self, filepath):
        """Load EMA history from a JSON file and restore current values."""
        with open(filepath, 'r') as f:
            self.history = json.load(f)
        for task in self.task_names:
            if self.history.get(task):
                self.values[task] = self.history[task][-1]
            else:
                self.values[task] = None