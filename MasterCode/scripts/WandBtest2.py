import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.loggers import WandbLogger

import wandb

# ─────────────────────────────────────────────
# Dummy dataset
# ─────────────────────────────────────────────
X = torch.randn(1000, 10)
y = torch.sum(X, dim=1, keepdim=True)  # simple regression target

dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32)

# ─────────────────────────────────────────────
# Simple Lightning model
# ─────────────────────────────────────────────
class SimpleModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # log to wandb
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# ─────────────────────────────────────────────
# WandB Logger
# ─────────────────────────────────────────────

print("starting logger")
wandb_logger = WandbLogger(
    project="wandb-lightning-test",
    log_model=False,
)
print("logger finished")
# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────
trainer = L.Trainer(
    max_epochs=5,
    logger=wandb_logger,
    log_every_n_steps=1,
)

# ─────────────────────────────────────────────
# Run training
# ─────────────────────────────────────────────
model = SimpleModel()
trainer.fit(model, train_loader)

print("training")
# Finish wandb run
wandb.finish()
print("finished")