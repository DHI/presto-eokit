# Maintainer: Walid Ghariani 
# Description:  PrestoLightningModule as a subclass from PyTorch Lightning


import lightning as L
import torch
import torch.nn as nn
from torch.optim import AdamW

from .optim_utils import get_layerwise_lr_decay


class PrestoLightningModule(L.LightningModule):
    def __init__(
        self, model, loss_fn, optimizer_config, scheduler_config=None, metrics=None
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": nn.ModuleDict(metric_dict)
                for phase, metric_dict in (metrics or {}).items()
            }
        )
        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def forward(self, x, dynamic_world, mask, latlons, month):
        return self.model(
            x=x, dynamic_world=dynamic_world, mask=mask, latlons=latlons, month=month
        )

    def _shared_step(self, batch, step_type):
        x, mask, dw, latlons, y, month = batch
        y_pred = self(x, dw, mask, latlons, month)
        loss = self.loss_fn(y_pred.squeeze(), y.squeeze())

        self.log(
            f"{step_type}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        metric_key = f"{step_type}_metrics"
        if metric_key in self.metrics:
            for name, fn in self.metrics[metric_key].items():
                value = fn(y_pred.squeeze(), y.squeeze())
                self.log(
                    f"{step_type}_{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx):
        x, mask, dw, latlons, month = batch
        y_pred = self(x, dw, mask, latlons, month)
        return y_pred

    def configure_optimizers(self):
        opt_cfg = self.optimizer_config
        wd = opt_cfg.get("weight_decay", 0.0)
        base_lr = opt_cfg.get("base_lr", opt_cfg.get("lr"))
        use_layerwise = opt_cfg.get("use_layerwise", False)

        if use_layerwise:
            param_groups = get_layerwise_lr_decay(
                self.model,
                base_lr=base_lr,
                decay_factor=opt_cfg.get("decay_factor", 0.9),
                weight_decay=wd,
            )
            optimizer = AdamW(param_groups)
        else:
            optimizer = AdamW(self.parameters(), lr=base_lr, weight_decay=wd)

        if self.scheduler_config:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **self.scheduler_config
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }
        return optimizer
