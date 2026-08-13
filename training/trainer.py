"""Shared training loop: Adam optimiser with a 1-cycle LR schedule
(Sec. 4.2's stated optimiser/schedule for both branches), used by all
three train_*.py entrypoints via a model-specific `forward_fn`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader


@dataclass
class OptimConfig:
    max_lr: float
    epochs: int
    batch_size: int


def build_optimizer_and_scheduler(model: nn.Module, config: OptimConfig, steps_per_epoch: int):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config.max_lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.max_lr,
        epochs=config.epochs,
        steps_per_epoch=max(steps_per_epoch, 1),
    )
    return optimizer, scheduler


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        forward_fn: Callable[[nn.Module, dict], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.forward_fn = forward_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            labels = batch["label"]

            embeddings = self.forward_fn(self.model, batch)
            loss = self.loss_fn(embeddings, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
        return total_loss / max(num_batches, 1)

    def fit(self, dataloader: DataLoader, epochs: int, checkpoint_path: Optional[str] = None, log_every: int = 1):
        history = []
        best_loss = float("inf")
        for epoch in range(1, epochs + 1):
            avg_loss = self.train_epoch(dataloader)
            history.append(avg_loss)
            if epoch % log_every == 0 or epoch == epochs:
                print(f"epoch {epoch}/{epochs}  loss={avg_loss:.4f}")
            if checkpoint_path is not None and avg_loss < best_loss:
                best_loss = avg_loss
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), checkpoint_path)
        return history
